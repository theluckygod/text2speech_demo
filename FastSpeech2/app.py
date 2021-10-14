import re
import os
import argparse
from string import punctuation

import torch
from underthesea.pipeline.word_tokenize import model
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
from viphoneme import vi2IPA_split

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from text import text_to_sequence, my_viphoneme

import yaml
import streamlit as st
from vinorm import TTSnorm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    'restore_step': 100000,
    'preprocess_config': './config/my_data/preprocess.yaml',
    'model_config': './config/my_data/model.yaml',
    'train_config': './config/my_data/train.yaml',
    'pitch_control': 1.0,
    'energy_control': 1.0
}

@st.cache
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


@st.cache
def preprocess_vietnamese(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = lambda s: my_viphoneme.get_my_viphoneme_list(my_viphoneme.get_cleaned_viphoneme_list(s))
    
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != "", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


# @st.cache
def synthesize(model, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


@st.cache(allow_output_mutation=True)
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


@st.cache
def load_conf():
    with open("config.yml", "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, yaml.Loader)
    return cfg
cfg = load_conf()

# Title
st.title("Vietnamese Text to Speech")

# Side bar
st.sidebar.subheader('Speaker')
speaker_values = list(range(cfg['conf_nof_speaker']['fastspeech2']))
speaker_id = st.sidebar.selectbox('', speaker_values, index=0)
        
st.sidebar.subheader('Speed')
speed_values = cfg["conf_values"]["speed"]
speed_default = cfg["conf_default"]["speed"]
speed = st.sidebar.selectbox('', speed_values, index=speed_values.index(speed_default))


default_value_text_area = cfg['conf_st_app']['default_value_text_area']
max_height_text_area = 150
default_audio_path = cfg['conf_st_app']['default_audio_path']
output_audio_path = cfg['conf_st_app']['output_audio_path']

# Content
sentence = st.text_area(label='Input your text here:',
                        value=default_value_text_area,
                        height=max_height_text_area) 

model, vocoder, configs = prepare_model(args)

if st.button("Generate audio"):
    flt_speed = {'Normal': 1.0, 'Slow': 1.5, 'Fast': 0.7}
    speed = flt_speed[speed]
   
    sentence=TTSnorm(sentence)
    print(sentence)

    preprocess_config = configs[0] 
    ids = raw_texts = [sentence[:100]]
    speakers = np.array([speaker_id])
    texts = np.array([preprocess_vietnamese(sentence, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args['pitch_control'], args['energy_control'], speed

    synthesize(model, configs, vocoder, batchs, control_values)

    output_audio_path = 'results/output.wav'
    st.audio(output_audio_path, format='audio/wav')