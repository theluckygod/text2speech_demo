FROM theluckygod/fastspeech2
USER root

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

COPY . /tts
WORKDIR /tts

CMD /bin/bash -c "python3 api.py"