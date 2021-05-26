import streamlit as st
import SessionState

import matplotlib
import matplotlib.pylab as plt

import sys
sys.path.append('./tacotron2/')
sys.path.append('./hifi-gan/')

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
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

from scipy.io.wavfile import write

import time


accent_default = 'Northern Accent'
speed_default = 'Normal'
sampling_rate_default = 16000
default_value_text_area='Đây là tiếng nói Việt Nam phát thanh từ Hà Nội thủ đô nước Cộng hòa Xã hội Chủ nghĩa Việt Nam.'
max_height_text_area = None
output_audio_path = 'output/output.wav'
default_audio_path = 'output/default.wav'


@st.cache
def init():
    print("\n--------------------------------------------------")
    print("initting...")
    hparams = create_hparams()
    hparams.sampling_rate = sampling_rate
    print(hparams.n_symbols)
    return hparams

def load_model_text2mel(accent, speed):
    ## load model text2mel
    checkpoint_path = "./tacotron2_checkpoints/checkpoint_53000"
    print(-1)
    print(hparams)
    model_text2mel = load_model(hparams)
    print(0)
    model_text2mel.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    print(1)
    _ = model_text2mel.cuda().eval().half()
    return model_text2mel
def load_model_mel2audio():
    ## load model mel2audio
    checkpoint_path = "./hifi-gan_checkpoints/g_02520000"
    config_file = "./hifi-gan_checkpoints/config.json"

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

@st.cache
def _load_model(accent, speed):
    print("\n--------------------------------------------------")
    print("load model...")

    try:
        model_text2mel = load_model_text2mel(accent, speed)
    except:
        print("--- fail to load model text2mel")
        torch.cuda.empty_cache()
        return
    print("--- load model text2mel successfully")
    try:
        model_mel2audio, denoiser = load_model_mel2audio()
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

def infer_mel2audio(model_text2mel, text):
    with torch.no_grad():
        x = mel_outputs_postnet
        if len(x.shape) < 3:  # for mel from vivos
            x = x.unsqueeze(0) 
        start_time = time.time() 
        y_g_hat = model_mel2audio(x)
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

@st.cache
def inference(model_text2mel, model_mel2audio, denoiser, text):
    print("\n--------------------------------------------------")
    print("inference...")
    print("input: " + text)
    
    # text2mel
    mel = infer_text2mel(model_text2mel, text)
    
    # mel2audio
    audio = infer_mel2audio(model_mel2audio, mel)
    
    # denoise
    audio = denoise_audio(denoiser, audio)

    return audio
    

# Side bar
st.sidebar.subheader('Accent')
accent_values = ('Northern Accent', 'Central Accent', 'Southern Accent')
accent = st.sidebar.selectbox('', accent_values, index=accent_values.index(accent_default))
        
st.sidebar.subheader('Speed')
speed_values = ('Slow', 'Normal', 'Fast')
speed = st.sidebar.selectbox('', speed_values, index=speed_values.index(speed_default))

st.sidebar.subheader('Sampling rate')
sampling_rate_values = (16000, 22050, 24000, 48000)
sampling_rate = st.sidebar.selectbox('', sampling_rate_values, index=sampling_rate_values.index(sampling_rate_default))

# Content
st.title("Text to speech")

sentence = st.text_area(label='Input your text here:',
                        value=default_value_text_area,
                        height=max_height_text_area) 
        

hparams = init()
model_text2mel, model_mel2audio, denoiser = _load_model(accent, speed)
session_state = SessionState.get(name='', _audio=None)

if st.button("Generate"):
    data = inference(model_text2mel, model_mel2audio, denoiser, sentence)
    # write array to file:
    write(output_audio_path, hparams.sampling_rate, data)
    print("Writing to " + output_audio_path)
    session_state._audio = True
    st.audio(output_audio_path, format='audio/wav')
else:
    if session_state._audio:
        st.audio(output_audio_path, format='audio/wav')
    else:
        st.audio(default_audio_path, format='audio/wav')
