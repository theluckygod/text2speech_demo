# Demo text to speech with streamlit (FastSpeech2 + hifi-gan) on Vietnamese Dataset

## Table of contents

1. [Introduction](#introduction)
2. [Pretrain models](#models)
3. [Quick start](#quickstart)
4. [Demo with our pretrained models](#demo)
5. [Citation](#citation)

## Introduction <a name="introduction"></a>

This repo demo text to speech with streamlit (FastSpeech2 + hifi-gan) or [colab](https://colab.research.google.com/drive/1h6oPgvswwTEqMDzSXEFi1qTIjyyQl0Er?usp=sharing#scrollTo=HCszbfi1JIoE)

## Pretrain models <a name="models"></a>
We use two pretrained models:
    
1. [Tacotron2](https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712786_student_hcmus_edu_vn/EdygQSs8Gh1PjRR1cGOfw1MBD8AOeJjtm8nSArg_Hr4tGA?e=PxN9q6\&download=1): a acoustic model
2. [Hifi-Gan](https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712786_student_hcmus_edu_vn/EY-UfGisD2VEqLxjabwqy60BJ9hGI0ByRIriaRvfrWC8qA?e=HVst2H\&download=1): a vocoder model

## Quick Start <a name="quickstart"></a>

[Open in Colab](https://colab.research.google.com/drive/1h6oPgvswwTEqMDzSXEFi1qTIjyyQl0Er?usp=sharing#scrollTo=HCszbfi1JIoE) or [Quick demo](https://colab.research.google.com/drive/1h6oPgvswwTEqMDzSXEFi1qTIjyyQl0Er?usp=sharing#scrollTo=HCszbfi1JIoE)

## Run demo <a name="demo"></a>

### Install packages and download pretrained models

    ./bash.sh

### Run API
    python api.py

### Run web demo

    streamlit run app.py

## Citation <a name="citation"></a>

    @INPROCEEDINGS{8461368,
    author      = {Shen, Jonathan and Pang, Ruoming and Weiss, Ron J. and Schuster, Mike and Jaitly, Navdeep and Yang, Zongheng and Chen, Zhifeng and Zhang, Yu and Wang, Yuxuan and Skerrv-Ryan, Rj and Saurous, Rif A. and Agiomvrgiannakis, Yannis and Wu, Yonghui},  
    booktitle   = {2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    title       = {Natural TTS Synthesis by Conditioning Wavenet on MEL Spectrogram Predictions},
    year        = {2018},
    pages       = {4779-4783},
    doi         = {10.1109/ICASSP.2018.8461368}
    }

    @article{Kong2020HiFiGANGA,
    title       = {HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis},
    author      = {Jungil Kong and Jaehyeon Kim and Jaekyoung Bae},
    journal     = {ArXiv},
    year        = {2020},
    volume      = {abs/2010.05646}
    }
