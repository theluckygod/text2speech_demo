from flask import Flask, request, send_from_directory, jsonify
from flask_restful import reqparse, abort, Api, Resource

from init_api import prepare_model, inference, load_conf
from scipy.io.wavfile import write

import numpy as np
import io
import json

import time
import threading
from queue import Empty, Queue

args = {
    'restore_step': 660000,
    'preprocess_config': './FastSpeech2/config/my_data/preprocess.yaml',
    'model_config': './FastSpeech2/config/my_data/model.yaml',
    'train_config': './FastSpeech2/config/my_data/train.yaml',
    'pitch_control': 1.0,
    'energy_control': 1.0
}

BATCH_SIZE = 1
BATCH_TIMEOUT = 0.01
CHECK_INTERVAL = 0.01
requests_queue = Queue()

app = Flask(__name__)
api = Api(app)

cfg = load_conf()
SPEAKERS = list(range(cfg['conf_nof_speaker']['fastspeech2']))
SPEAKER_DEFAULT = 0
SPEED_VALUES = cfg["conf_values"]["speed"]
SPEED_DEFAULT = cfg["conf_default"]["speed"]
SAMPLING_RATE_VALUES =  cfg["conf_values"]["sampling_rate"]
SAMPLING_RATE_DEFAULT = cfg["conf_default"]["sampling_rate"]
output_audio_path = cfg['conf_st_app']['output_audio_path']
VALID_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Imk2bEdrM0ZaenhSY1ViMkMzbkVRN3N5SEpsWSIsImtpZCI6Imk2bEdrM0ZaenhSY1ViMkMzbkVRN3N5SEpsWSJ9.eyJhdWQiOiJlZjFkYTlkNC1mZjc3LTRjM2UtYTAwNS04NDBjM2Y4MzA3NDUiLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC9mYTE1ZDY5Mi1lOWM3LTQ0NjAtYTc0My0yOWYyOTUyMjIyOS8iLCJpYXQiOjE1MzcyMzMxMDYsIm5iZiI6MTUzNzIzMzEwNiwiZXhwIjoxNTM3MjM3MDA2LCJhY3IiOiIxIiwiYWlvIjoiQVhRQWkvOElBQUFBRm0rRS9RVEcrZ0ZuVnhMaldkdzhLKzYxQUdyU091TU1GNmViYU1qN1hPM0libUQzZkdtck95RCtOdlp5R24yVmFUL2tES1h3NE1JaHJnR1ZxNkJuOHdMWG9UMUxrSVorRnpRVmtKUFBMUU9WNEtjWHFTbENWUERTL0RpQ0RnRTIyMlRJbU12V05hRU1hVU9Uc0lHdlRRPT0iLCJhbXIiOlsid2lhIl0sImFwcGlkIjoiNzVkYmU3N2YtMTBhMy00ZTU5LTg1ZmQtOGMxMjc1NDRmMTdjIiwiYXBwaWRhY3IiOiIwIiwiZW1haWwiOiJBYmVMaUBtaWNyb3NvZnQuY29tIiwiZmFtaWx5X25hbWUiOiJMaW5jb2xuIiwiZ2l2ZW5fbmFtZSI6IkFiZSAoTVNGVCkiLCJpZHAiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC83MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMjIyNDcvIiwiaXBhZGRyIjoiMjIyLjIyMi4yMjIuMjIiLCJuYW1lIjoiYWJlbGkiLCJvaWQiOiIwMjIyM2I2Yi1hYTFkLTQyZDQtOWVjMC0xYjJiYjkxOTQ0MzgiLCJyaCI6IkkiLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzdWIiOiJsM19yb0lTUVUyMjJiVUxTOXlpMmswWHBxcE9pTXo1SDNaQUNvMUdlWEEiLCJ0aWQiOiJmYTE1ZDY5Mi1lOWM3LTQ0NjAtYTc0My0yOWYyOTU2ZmQ0MjkiLCJ1bmlxdWVfbmFtZSI6ImFiZWxpQG1pY3Jvc29mdC5jb20iLCJ1dGkiOiJGVnNHeFlYSTMwLVR1aWt1dVVvRkFBIiwidmVyIjoiMS4wIn0.D3H6pMUtQnoJAGq6AHd'
User = [{
  'email': 'admin@vlsp.com.vn',
  'password': 'admin'
}]

model_text2mel, model_mel2audio, configs = prepare_model(args)

def abort_if_config_doesnt_exist(token, speaker, speed, sampling_rate):
    if token != VALID_TOKEN:
        abort(404, message='Not valid token!')

    check_speaker, check_speed, check_sr = True, True, True
    if int(speaker) not in SPEAKERS:
        check_speaker = False
    # if speed not in SPEED_VALUES:
    #     check_speed = False
    if sampling_rate not in SAMPLING_RATE_VALUES:
        check_sr = False    

    if check_speaker == False or check_speed == False or check_sr == False:
        abort(404, message="Config not match {}".format(
            {"check_speaker": check_speaker, "check_speed": check_speed, "check_sr": check_sr}))

def handle_requests():
    """
    This function handle requests.
    """
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
        speakers = [request['input']['speaker_id'] for request in requests_batch]
        speeds = [request['input']['rate'] for request in requests_batch]
        sampling_rates = [request['input']['sr'] for request in requests_batch]
        # try:
        #     sentence = texts[0]
        #     speaker = int(speakers[0])
        #     speed = speeds[0]
        #     sampling_rate = sampling_rates[0]
        #     request = requests_batch[0]

        #     data = inference(args,
        #                     cfg,
        #                     configs[0],
        #                     model_text2mel, 
        #                     model_mel2audio, 
        #                     None, 
        #                     sentence, 
        #                     speaker, 
        #                     speed, 
        #                     sampling_rate)
        #     request['output'] = data
        # except:
        #     request['output'] = "Fail"
        #     continue
        
        sentence = texts[0]
        speaker = int(speakers[0])
        speed = speeds[0]
        sampling_rate = sampling_rates[0]
        request = requests_batch[0]

        data = inference(args,
                        cfg,
                        configs[0],
                        model_text2mel, 
                        model_mel2audio, 
                        None, 
                        sentence, 
                        speaker, 
                        speed, 
                        sampling_rate)
        request['output'] = data

threading.Thread(target=handle_requests).start()


# Todo
class Login(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('email', required=True, help='Email')
        parser.add_argument('password', required=True, help='Mat khau')
        
        args = parser.parse_args()
        email, password = args['email'], args['password']

        # Check account
        access_token = None
        status = 0
        try:
          if (User[0]['email'] == email and User[0]['password'] == password):
            access_token = VALID_TOKEN
            status = 1
        except:
          status = 0

        return jsonify({"status": status, "result":{'access_token': access_token}})

class Inference_e2e(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('access-token', help='Access Token', type=str, location='headers')
        parser.add_argument('text', required=True, help='Text input')
        parser.add_argument('speaker_id', default='0', help='Speaker of speech')
        parser.add_argument('rate', default=1.0, type=float, help='Speed of speech')
        parser.add_argument('sr', default=22050, type=int, help='Sampling rate of speech')

        args = parser.parse_args()
        token, sentence, speaker, speed, sampling_rate = args['access-token'], args['text'], args['speaker_id'], args['rate'], args['sr']
        
        abort_if_config_doesnt_exist(token, speaker, speed, sampling_rate)

        crequest = {'input': args, 'time': time.time()}
        requests_queue.put(crequest)
        
        while 'output' not in crequest:
            time.sleep(CHECK_INTERVAL)

        data = crequest['output']

        if not isinstance(data, np.ndarray):
            return abort(403, message='Fauty Inference!') # Fail to infer
            
        data = data.tolist()
        write('output/output.wav', 22050, np.array(data, np.int16))
        return send_from_directory('output', filename='output.wav', attachment_filename='output.wav', as_attachment=True)


##
## Actually setup the Api resource routing here
##
api.add_resource(Login, '/login')
api.add_resource(Inference_e2e, '/tts')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
