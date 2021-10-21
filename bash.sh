#!/bin/bash

# install packages
pip install -r requirements.txt

# Make dir
mkdir ./FastSpeech2/ckpt

# download pretrained models
wget -O ./FastSpeech2/ckpt/300000.pth.tar https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712791_student_hcmus_edu_vn/Ea_RN9BYVKJNgW2fpr1tXx8B514SWxDwm2SUdmrf85HgxA?e=vDdZ2W\&download=1# model pretrained fastspeech
wget -O ./FastSpeech2/hifigan/generator_universal.pth.tar https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712791_student_hcmus_edu_vn/Ec6TjvI8oIZNrSfqlKM8xJIB13E_z8QNSVLuz7kySBeoYQ?e=xlPb99\&download=1# model pretrained hifi-gan
