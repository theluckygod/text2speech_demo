import os
import sys
import time
import glob
# sys.path.insert(0,'/home/ubuntu/TTS/text-to-speech/FastSpeech2')
import threading
from queue import Empty, Queue
from pydub import AudioSegment
from utils.text_breaker import breaker
# from model_prediction import T2S_Runner
from underthesea import sent_tokenize, word_tokenize
from flask import Flask, redirect, url_for, request
from torch.multiprocessing import Process,set_start_method, Pool

from app import load_model, inference

# import scipy.io.wavfile.read as wavread
set_start_method('spawn', force = True)
# Add global config and variable
BATCH_SIZE = 30
BATCH_TIMEOUT = 0.01
CHECK_INTERVAL = 0.01
MAX_PROCESS = 2
preprocess_config = "./config/LJSpeech/preprocess.yaml"
model_config = './config/LJSpeech/model.yaml'
train_config = './config/LJSpeech/train.yaml'
restore_step = 900000
requests_queue = Queue()
app = Flask(__name__)

def handle_requests_by_batch_with_break_sentence():
    """
    This function is control and create batch of requests.
    """
    model = T2S_Runner(preprocess_config, model_config, train_config, restore_step)
    while True:
        requests_batch = []
        while not (
            len(requests_batch) > BATCH_SIZE or
            (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue
        texts = [request['input']['text'] for request in requests_batch]
        texts_split = [breaker(t) for t in texts]
        texts_split_len = [len(t) for t in texts_split]
        texts = []
        for t in texts_split:
            texts.extend([i+" " for i in t])
        IDs = []
        speaker_ids = []
        for request, l in zip (requests_batch, texts_split_len):
            ids = ['%s_%s'%(request['time'],i) for i in range(l)]
            IDs.extend(ids)
            speaker_ids.extend([request['input']['speaker_id']]*l)
        pitch_controls = requests_batch[0]['input']['pitch_control']
        energy_controls = requests_batch[0]['input']['energy_control']
        duration_controls = requests_batch[0]['input']['duration_control']
        batch_outputs = model.batch_prediction(IDs, texts, speaker_ids, pitch_controls, energy_controls, duration_controls)
        IDs = [request['time'] for request in requests_batch]
        for id, request in zip(IDs,requests_batch):
            files = sorted(glob.glob('./output/result/%s_*.wav'%id))
            combined_sounds = AudioSegment.from_wav(files[0])
            os.remove(files[0])
            for file in files[1:]:
                sound = AudioSegment.from_wav(file)
                combined_sounds = combined_sounds + sound
                os.remove(file)
            combined_sounds.export('./output/result/%s.wav'%id, format="wav")
            request['output'] = "GOOD"

threading.Thread(target=handle_requests_by_batch_with_break_sentence).start()


@app.route('/predict', methods=['POST'])
def predict():
    crequest = {'input': request.json, 'time': time.time()}
    requests_queue.put(crequest)

    while 'output' not in crequest:
        time.sleep(CHECK_INTERVAL)

    return {'predictions': crequest['output']}


if __name__ == "__main__":
    app.run(debug = True)