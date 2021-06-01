import yaml

import sys
sys.path.append('./tacotron2/')
sys.path.append('./hifi-gan/')

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model as _load_model
from text import text_to_sequence

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
from inference_e2e import load_checkpoint

from vinorm import TTSnorm
from underthesea import sent_tokenize, dependency_parse


import time

silence_sign = {
    '\n': 130,
    '.': 60,
    ',': 30,
    '': 0
}



def load_conf():
    with open("config.yml", "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, yaml.Loader)
    return cfg

def init(cfg):
    print("\n--------------------------------------------------")
    print("initting...")
    hparams = create_hparams()
    hparams.sampling_rate = cfg["conf_default"]["sampling_rate"]
    print(hparams.n_symbols)
    return hparams
def load_model_text2mel(cfg):
    ## load model text2mel
    checkpoint_path = cfg["conf_model"]["tacotron2"]["checkpoint"]
    hparams = init(cfg)
    model_text2mel = _load_model(hparams)
    model_text2mel.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model_text2mel.cuda().eval().half()
    return model_text2mel
def load_model_mel2audio(cfg):
    ## load model mel2audio
    checkpoint_path = cfg["conf_model"]["hifi-gan"]["checkpoint"]
    config_file = cfg["conf_model"]["hifi-gan"]["config"]

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    assert(torch.cuda.is_available())
    device = 'cuda'
    torch.cuda.manual_seed(h.seed)

    model_mel2audio = Generator(h).to(device)
    state_dict_g = load_checkpoint(checkpoint_path, device)
    model_mel2audio.load_state_dict(state_dict_g['generator'])
    model_mel2audio.eval().half()
    model_mel2audio.remove_weight_norm()

    denoiser = None
    return model_mel2audio, denoiser

def load_model(cfg):
    print("\n--------------------------------------------------")
    print("load model...")

    try:
        model_text2mel = load_model_text2mel(cfg)
    except:
        print("--- fail to load model text2mel")
        torch.cuda.empty_cache()
        return
    print("--- load model text2mel successfully")
    try:
        model_mel2audio, denoiser = load_model_mel2audio(cfg)
    except:
        print("--- fail to load model mel2audio")
        torch.cuda.empty_cache()
        return
    print("--- load model mel2audio successfully")

    return model_text2mel, model_mel2audio, denoiser

def infer_text2mel(model_text2mel, text):
    # preprocessing text
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    # text2mel
    start_time = time.time() 
    mel_outputs, mel_outputs_postnet, _, alignments = model_text2mel.inference(sequence)
    print("--- text2mel: %s seconds ---" % (time.time() - start_time))

    return mel_outputs_postnet

def infer_mel2audio(model_mel2audio, mel):
    with torch.no_grad():
        if len(mel.shape) < 3:  # for mel from vivos
            mel = mel.unsqueeze(0) 
        start_time = time.time() 
        y_g_hat = model_mel2audio(mel)
        print("--- mel2audio: %s seconds ---" % (time.time() - start_time))    
        audio = y_g_hat * MAX_WAV_VALUE

    return audio

def denoise_audio(denoiser, audio):
    if denoiser == None:
        data = audio.cpu().numpy().astype('int16')
    else:
        audio_denoised = denoiser(audio, strength=0.01)[:, 0]
        start_time = time.time()
        data = audio_denoised.cpu().numpy()
        print("--- mel2audio: %s seconds ---" % (time.time() - start_time))

    return np.squeeze(data)

def paragraph_sementation(text):
    text_list, silence_mark = split_sentence_with_character(text, "\n")
    return text_list, ["\n"] + silence_mark
    
def norm_text(text_list):
    for i in range(len(text_list)):
        text_list[i] = TTSnorm(text_list[i], punc = False, unknown = False, lower = True, rule = False)
    return text_list

def split_sentence_with_character(text, character):
    text_list = text.split(character)
    selected_texts = []
    for s in text_list:
        if len(s) > 0:
            selected_texts.append(s)
    return selected_texts, [character] * (len(selected_texts) - 1)

def sentence_segmentation(text_list, silence_mark):
    selected_texts = []
    si_mark = []
    for i in range(len(text_list)):
        t_texts = sent_tokenize(text_list[i])
        t_mark = [silence_mark[i]] + ['.'] * (len(t_texts) - 1)

        selected_texts = selected_texts + t_texts
        si_mark = si_mark + t_mark
    return selected_texts, si_mark

def split_long_sentence(text_list, silence_mark):
    # puntc ,
    selected_texts = []
    si_mark = []
    for i in range(len(text_list)):
        t_texts, t_mark = split_sentence_with_character(text_list[i], ',')
        t_mark = [silence_mark[i]] + t_mark

        selected_texts = selected_texts + t_texts
        si_mark = si_mark + t_mark
    return selected_texts, si_mark
    
def text_preprocessing(text):
    # paragraph segmentation
    text_list, silence_mark = paragraph_sementation(text)

    # norm
    text_list = norm_text(text_list)
    
    # sentence segmentation
    text_list, silence_mark = sentence_segmentation(text_list, silence_mark)
    
    # split long sentance
    text_list, silence_mark = split_long_sentence(text_list, silence_mark)
    if len(silence_mark) > 0:
        silence_mark[0] = ''
    return text_list, silence_mark

def inference(model_text2mel, model_mel2audio, denoiser, text, accent, speed, sampling_rate):
    print("\n--------------------------------------------------")
    print("inference...")
    print("input: " + text)

    texts, si_mark = text_preprocessing(text)

    # TODO
    # Accent, speed, sampling_rate?

    # text2mel
    mel = None
    for i in range(len(texts)):
        if isinstance(mel, torch.Tensor):
            temp_mel = infer_text2mel(model_text2mel, texts[i])
            if si_mark[i] != '':
                _si_mel = mel[:, :, -1].unsqueeze(-1)
                _si_mel = _si_mel.repeat_interleave(silence_sign[si_mark[i]], dim=-1)
                mel = torch.cat((mel, _si_mel, temp_mel), 2)
            else:
                mel = torch.cat((mel, temp_mel), 2)

        else:
            mel = infer_text2mel(model_text2mel, texts[i])
    # mel2audio
    audio = infer_mel2audio(model_mel2audio, mel)
    # denoise
    audio = denoise_audio(denoiser, audio)
    return audio