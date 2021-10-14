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

	@@ -38,30 +47,68 @@ def abort_if_config_doesnt_exist(accent, speed, sampling_rate):
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

            data = inference(model_text2mel, 
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
