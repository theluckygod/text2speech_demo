from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource

from init_api import load_model, inference, load_conf
from scipy.io.wavfile import write
import numpy as np

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
        print(ACCENTS)
        print(accent)
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


# Todo
class Inference_e2e(Resource):
    def post(self):
        args = parser.parse_args()
        sentence, accent, speed, sampling_rate = args['text'], args['accent'], args['speed'], args['sr']
        abort_if_config_doesnt_exist(accent, speed, sampling_rate)

        try:
            data = inference(model_text2mel, 
                            model_mel2audio, 
                            denoiser, 
                            sentence, 
                            accent, 
                            speed, 
                            sampling_rate)
        except:
            print("Fail to infer")
            return None, 403

        if isinstance(data, np.ndarray):
            data = data.tolist()
        return {"data": data, "sr": sampling_rate}, 201


##
## Actually setup the Api resource routing here
##
api.add_resource(Inference_e2e, '/inference')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)