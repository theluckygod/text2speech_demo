import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from utils import (
    load_wav_to_torch,
    load_filepaths_and_text,
    #guide_attention_fast,
    to_gpu,
)

import argparse

from hparams import create_hparams
from model import Tacotron2
from torch.utils.data.distributed import DistributedSampler
from train import load_model, load_checkpoint, init_distributed
from tqdm import tqdm
from data_utils import TextMelLoader, TextMelCollate


def prepare_dataloader(hparams, file_list):
    # Get data, data loaders and collate function ready
    eval_set = TextMelLoaderWithPath(file_list, hparams)
    collate_fn = TextMelCollateWithPath(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(eval_set)
        shuffle = False
    else:
        train_sampler = None
        shuffle = False

    eval_loader = DataLoader(
        eval_set,
        num_workers=1,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=hparams.batch_size,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return eval_loader


def parse_batch(batch):
    (
        text_padded,
        input_lengths,
        mel_padded,
        gate_padded,
        output_lengths,
        ctc_text,
        ctc_text_lengths,
        #guide_mask,
        fnames,
    ) = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    ctc_text = to_gpu(ctc_text).long()
    ctc_text_lengths = to_gpu(ctc_text_lengths).long()
    #guide_mask = to_gpu(guide_mask).float()

    return (
        (
            text_padded,
            input_lengths,
            mel_padded,
            max_len,
            output_lengths,
            ctc_text,
            ctc_text_lengths,
        ),
        (fnames, mel_padded, gate_padded),
    )


class TextMelLoaderWithPath(TextMelLoader):
    def __init__(self, audiopaths_and_text, hparams):
        super().__init__(audiopaths_and_text, hparams)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = (
            audiopath_and_text[0],
            audiopath_and_text[1].strip(),
        )
        text, ctc_text = self.get_text(text)
        mel = self.get_mel(os.path.join(audiopath))
        # guide_mask = torch.FloatTensor(
        #     guide_attention_fast(len(text), mel.shape[-1], 450, 3000)
        # )
        return (text, ctc_text, mel, audiopath)


class TextMelCollateWithPath(TextMelCollate):
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        super().__init__(n_frames_per_step)

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        max_ctc_txt_len = max([len(x[1]) for x in batch])
        ctc_text_paded = torch.LongTensor(len(batch), max_ctc_txt_len)
        ctc_text_paded.zero_()
        ctc_text_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            ctc_text = batch[ids_sorted_decreasing[i]][1]
            ctc_text_paded[i, : ctc_text.size(0)] = ctc_text
            ctc_text_lengths[i] = ctc_text.size(0)

        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

        guide_padded = torch.FloatTensor(len(batch), 450, 3000)
        guide_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            guide = batch[ids_sorted_decreasing[i]][3]
            #guide_padded[i, :, :] = guide

        fnames = [
            batch[ids_sorted_decreasing[i]][3]
            for i in range(len(ids_sorted_decreasing))
        ]

        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            ctc_text_paded,
            ctc_text_lengths,
            #guide_padded,
            fnames,
        )


def extract_mels_teacher_forcing(
    output_directory,
    checkpoint_path,
    hparams,
    file_list,
    n_gpus,
    rank,
    group_name,
    extract_type="mels",
):
    device = torch.device("cuda:{:d}".format(rank))

    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    eval_loader = prepare_dataloader(hparams, file_list)
    model = load_model(hparams)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    model.eval()
    # if hparams.fp16_run:
    #     model.half()
    for batch in tqdm(eval_loader):
        x, y = parse_batch(batch)
        with torch.no_grad():
            y_pred = model(x)

        if extract_type == "mels":
            for res, fname, out_length in zip(y_pred[2], y[0], x[4]):
                speaker_name, fname = fname.split('/')[-1].split('.')[0].split('_')[0], fname.split('/')[-1].split('.')[0]
                np.save(
                    os.path.join(output_directory, speaker_name, fname + ".npy"),
                    res.cpu().numpy()[:, :out_length],
                )
        elif extract_type == "alignments":
            for alignment, fname, seq_len, out_length in zip(
                y_pred[4], y[0], x[1], x[4]
            ):
                alignment = alignment.T[:seq_len, :out_length]
                np.save(
                    os.path.join(output_directory, fname + ".npy"),
                    np.bincount(
                        np.argmax(alignment.cpu().numpy(), axis=0),
                        minlength=alignment.shape[0],
                    ),
                )
        else:
            raise Exception(f"Extracting {extract_type} is not supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        help="directory to save extracted features",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["mels", "alignments"],
        help="Whether to extract mels or alignments",
    )
    parser.add_argument("--filelist", type=str, help="comma separated name=value pairs")
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "--n_gpus", type=int, default=3, required=False, help="number of gpus"
    )
    parser.add_argument(
        "--rank", type=int, default=0, required=False, help="rank of current gpu"
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default="group_name",
        required=False,
        help="Distributed group name",
    )
    parser.add_argument(
        "--hparams", type=str, required=False, help="comma separated name=value pairs"
    )

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled

    os.makedirs(args.output_directory, exist_ok=True)

    extract_mels_teacher_forcing(
        args.output_directory,
        args.checkpoint_path,
        hparams,
        args.filelist,
        args.n_gpus,
        args.rank,
        args.group_name,
        args.type,
    )

