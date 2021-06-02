from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource

from init_api import load_model, inference, load_conf
from scipy.io.wavfile import write
import numpy as np

import time
import threading
from queue import Empty, Queue

BATCH_SIZE = 1
BATCH_TIMEOUT = 0.01
CHECK_INTERVAL = 0.01
requests_queue = Queue()

app = Flask(__name__)
api = Api(app)

cfg = load_conf()
ACCENTS = cfg["conf_values"]["accent"]
ACCENTS_DEFAULT = cfg["conf_default"]["accent"]
SPEED_VALUES = cfg["conf_values"]["speed"]
SPEED_DEFAULT = cfg["conf_default"]["speed"]
SAMPLING_RATE_VALUES =  cfg["conf_values"]["sampling_rate"]
SAMPLING_RATE_DEFAULT = cfg["conf_default"]["sampling_rate"]

model_text2mel, model_mel2audio, denoiser = load_model(cfg)


def abort_if_config_doesnt_exist(accent, speed, sampling_rate):
    check_accent, check_speed, check_sr = True, True, True
    if accent not in ACCENTS:
        check_accent = False
    if speed not in SPEED_VALUES:
        check_speed = False
    if sampling_rate not in SAMPLING_RATE_VALUES:
        check_sr = False    

    if check_accent == False or check_speed == False or check_sr == False:
        abort(404, message="Config not match {}".format(
            {"check_accent": check_accent, "check_speed": check_speed, "check_sr": check_sr}))

parser = reqparse.RequestParser()
parser.add_argument('text', help='Text input')
parser.add_argument('accent', help='Accent of speech')
parser.add_argument('speed', help='Speech of speech')
parser.add_argument('sr', type=int, help='Sampling rate of speech')

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
        accents = [request['input']['accent'] for request in requests_batch]
        speeds = [request['input']['speed'] for request in requests_batch]
        sampling_rates = [request['input']['sr'] for request in requests_batch]

        try:
            sentence = texts[0]
            accent = accents[0]
            speed = speeds[0]
            sampling_rate = sampling_rates[0]
            request = requests_batch[0]

            data = inference(cfg, 
                            model_text2mel, 
                            model_mel2audio, 
                            denoiser, 
                            sentence, 
                            accent, 
                            speed, 
                            sampling_rate)
            request['output'] = data
        except:
            request['output'] = "Fail"
            continue

threading.Thread(target=handle_requests).start()


# Todo
class Inference_e2e(Resource):
    def post(self):
        args = parser.parse_args()
        sentence, accent, speed, sampling_rate = args['text'], args['accent'], args['speed'], args['sr']
        abort_if_config_doesnt_exist(accent, speed, sampling_rate)

        crequest = {'input': args, 'time': time.time()}

        requests_queue.put(crequest)

        while 'output' not in crequest:
            time.sleep(CHECK_INTERVAL)

        data = crequest['output']

        if not isinstance(data, np.ndarray):
            return {"data": None, "sr": sampling_rate}, 403 # Fail to infer
            
        data = data.tolist()
        return {"data": data, "sr": sampling_rate}, 201 # infer successfully

##
## Actually setup the Api resource routing here
##
api.add_resource(Inference_e2e, '/inference')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)