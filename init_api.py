import yaml

import sys
sys.path.append('./FastSpeech2/')
sys.path.append('./hifi-gan/')

import numpy as np
import torch
import re
from string import punctuation

from underthesea.pipeline.word_tokenize import model
from g2p_en import G2p
from pypinyin import pinyin, Style
from viphoneme import vi2IPA_split

from FastSpeech2.text import text_to_sequence, my_viphoneme

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

from utils.model import get_model, get_vocoder
from utils.tools import to_device, pad_1D, pad_2D

from vinorm import TTSnorm
from underthesea import sent_tokenize, dependency_parse
import math

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def prepare_model(args):
    # Read Config
    preprocess_config = yaml.load(
        open(args['preprocess_config'], "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args['model_config'], "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args['train_config'], "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args['restore_step'], configs, device, train=False)
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    return model, vocoder, configs

def infer_text2mel(preprocess_config, model_text2mel, speaker, sequence, src_lens, max_src_len, control_values):
    # preprocessing text
    pitch_control, energy_control, duration_control = control_values
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    # text2mel
    start_time = time.time()
    speaker = torch.from_numpy(np.array(speaker)).long().to(device)
    sequence = sequence.long().to(device)
    src_lens = torch.from_numpy(src_lens).to(device)
    #max_src_len = to_device(max_src_len, device)
    with torch.no_grad():
      output = model_text2mel(
            speaker, 
            sequence, 
            src_lens, 
            max_src_len, 
            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control)
    
    print("--- text2mel: %s seconds ---" % (time.time() - start_time))

    mel_outputs_postnet = output[1]
    lengths = output[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    return mel_outputs_postnet, lengths

def infer_mel2audio(model_mel2audio, mels):
    with torch.no_grad():
        if len(mels.shape) < 3:  # for mel from vivos
            mels = mels.unsqueeze(0) 
        start_time = time.time() 
        y_g_hat = model_mel2audio(mels)
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

def norm_begin_end_text(text_list, silence_mark):
    selected_texts = []
    si_mark = []
    for i in range(len(text_list)):
        text = text_list[i]
        while True:
            if len(text) > 0 and text[-1] in ['.', ',', ' ', '\n']:
                text = text[:-1]
            else:
                break
        while True:
            if len(text) > 0 and text[0] in ['.', ',', ' ', '\n']:
                text = text[1:]
            else:
                break
        
        if len(text) > 0:
            text = text + '.'
            selected_texts.append(text)
            si_mark.append(silence_mark[i])
    return selected_texts, si_mark

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

def segment_sentence(text, factor):
    tokens = dependency_parse(text)
    sentence = [token[0] for token in tokens]

    s = ''
    for i in range(factor):
        if i > 0:
            s += ', '
        s += ' '.join(sentence[int(len(sentence) / factor) * i:int(len(sentence) / factor) * (i + 1)])
    return s

def split_long_sentence(cfg, text_list, silence_mark):
    # puntc ,
    selected_texts = []
    si_mark = []
    for i in range(len(text_list)):
        t_texts, t_mark = split_sentence_with_character(text_list[i], ',')
        t_mark = [silence_mark[i]] + t_mark

        selected_texts = selected_texts + t_texts
        si_mark = si_mark + t_mark

    text_list = selected_texts
    silence_mark = si_mark
    selected_texts = []
    si_mark = []
    max_len = cfg["conf_model"]["text_preprocessing"]["max_len_sentence"]

    for i in range(len(text_list)):
        if len(text_list[i]) > max_len:
            factor = math.ceil(len(text_list[i]) / max_len)

            segmented_sentence = segment_sentence(text_list[i], factor)

            t_texts, t_mark = split_sentence_with_character(segmented_sentence, ',')
            t_mark = [silence_mark[i]] + t_mark

            selected_texts = selected_texts + t_texts
            si_mark = si_mark + t_mark
        else:
            selected_texts.append(text_list[i])
            si_mark.append(silence_mark[i])

    dependency_parse
    return selected_texts, si_mark

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def text_preprocessing(cfg, text):
    # paragraph segmentation
    text_list, silence_mark = paragraph_sementation(text)
    
    # sentence segmentation
    text_list, silence_mark = sentence_segmentation(text_list, silence_mark)
    
    # split long sentance
    text_list, silence_mark = split_long_sentence(cfg, text_list, silence_mark)

    # norm
    text_list = norm_text(text_list)

    # re norm
    text_list, silence_mark = norm_begin_end_text(text_list, silence_mark)
    
    if len(silence_mark) > 0:
        silence_mark[0] = ''

    return text_list, silence_mark

def preprocess_vietnamese(text, preprocess_config, cfg):
    
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    text_list, si_mark = text_preprocessing(cfg, text)
    selected_sequences = []
    g2p = lambda s: my_viphoneme.get_my_viphoneme_list(my_viphoneme.get_cleaned_viphoneme_list(s))

    for sub_text in text_list:
      sub_text = sub_text.rstrip(punctuation)

      phones = []
      words = re.split(r"([,;.\-\?\!\s+])", sub_text)
      for w in words:
          if w.lower() in lexicon:
              phones += lexicon[w.lower()]
          else:
              phones += list(filter(lambda p: p != "", g2p(w)))
      phones = "{" + "}{".join(phones) + "}"
      phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
      phones = phones.replace("}{", " ")

      sequence = np.array(
          text_to_sequence(
              phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
          )
      )
      selected_sequences.append(sequence)

    return selected_sequences, si_mark

def inference(args, cfg, preprocess_config, model_text2mel, model_mel2audio, denoiser, text, speaker, speed, sampling_rate):
    print("\n--------------------------------------------------")
    print("inference...")
    print("input: " + text)

    start_time = time.time() 
    sequences, si_mark = preprocess_vietnamese(text, preprocess_config, cfg)
    print("--- preprocessing: %s seconds ---" % (time.time() - start_time))      

    # TODO
    # Accent, speed, sampling_rate?
    flt_speed = {'Normal': 1.0, 'Slow': 1.5, 'Fast': 0.7}
    speed = flt_speed[speed]

    # text2mel
    audio = None
    control_values = args['pitch_control'], args['energy_control'], speed

    # Infer batch
    sequence_lens = np.array([len(sequences[i]) for i in range(len(sequences))])
    sequences = pad_1D(sequences)
    sequences = np.array(sequences)
    speakers = [speaker] * len(sequences)
    mels, lengths = infer_text2mel(preprocess_config, model_text2mel, speakers, sequences, sequence_lens, max(sequence_lens), control_values)
    mels = mels.permute(0,2,1)

    # mel2audio
    temp_audio = infer_mel2audio(model_mel2audio, mels)
    audio = temp_audio[0,:,:lengths[0]].unsqueeze(0)

    # post-processing (concat audio)
    start_time = time.time() 
    for i in range(1, len(temp_audio)):
        if si_mark[i] != '':
            _si_audio = audio[:, :, -1].unsqueeze(-1)
            _si_audio = _si_audio.repeat_interleave(silence_sign[si_mark[i]] * 256, dim=-1)
            audio = torch.cat((audio, _si_audio, temp_audio[i,:,:lengths[i]].unsqueeze(0)), 2)
        else:
            audio = torch.cat((audio, temp_audio[i,:,:lengths[i]].unsqueeze(0)), 2)
    
    # denoise
    audio = denoise_audio(denoiser, audio)
    print("--- post-processing: %s seconds ---" % (time.time() - start_time))
    return audio
