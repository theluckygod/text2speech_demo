#!/bin/bash

# install packages
pip install -r requirements.txt

# Make dir
mkdir ./FastSpeech2/ckpt

# download pretrained models
wget -O ./FastSpeech2/ckpt/100000.pth.tar https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712791_student_hcmus_edu_vn/Eddrt4TbzZBNmmsDmUTsx1cBKl83DN7gC1sRlo45TLvKtg?e=jjaFwi\&download=1
wget -O ./FastSpeech2/hifigan/generator_universal.pth.tar https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712791_student_hcmus_edu_vn/Ec_tC7hVcFVJlTgOfXYDkjEBuvm_f4BhqEwtQc70k0hCHQ?e=Sn5p4g\&download=1
# unzip tacotron2_checkpoints.zip
# unzip hifi-gan_checkpoints.zip
