#!/bin/bash

# install packages
pip install -r requirements.txt

# Make dir
mkdir ./FastSpeech2/ckpt

# download pretrained models
wget -O ./FastSpeech2/ckpt/660000.pth.tar https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712791_student_hcmus_edu_vn/Ee17rI0_7_1PlLGS2XNZaokBeWMRsKQBmf5Z8-0bwdZs2w?e=x7Zkcb\&download=1# model pretrained fastspeech
wget -O ./FastSpeech2/hifigan/generator_universal.pth.tar https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712791_student_hcmus_edu_vn/Ebas_mFP3apMmxA42MNhvbsBbYrMwPERMcYBjU-ucDouQw?e=VQ9XkG\&download=1# model pretrained hifi-gan
