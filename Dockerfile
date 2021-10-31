FROM theluckygod/fastspeech2
USER root

RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN wget -O /root/.cache/torch/hub/checkpoints/vi-dp-v1a1.zip https://github.com/undertheseanlp/underthesea/releases/download/v1.3-resources/vi-dp-v1a1.zip

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt
RUN pip install --pre torch==1.6.0 torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html


COPY . /tts
WORKDIR /tts

CMD /bin/bash -c "python3 api.py"