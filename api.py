import streamlit as st
from scipy.io.wavfile import write
from requests import post
import numpy as np
import yaml
import io
import json

@st.cache
def load_conf():
    with open("config.yml", "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, yaml.Loader)
    return cfg
cfg = load_conf()
    
# # Side bar
# st.sidebar.subheader('Accent')
# accent_values = cfg["conf_values"]["accent"]
# accent_default = cfg["conf_default"]["accent"]
# accent = st.sidebar.selectbox('', accent_values, index=accent_values.index(accent_default))

# Side bar
st.sidebar.subheader('Speaker')
speaker_values = list(range(cfg['conf_nof_speaker']['fastspeech2']))
speaker_id = st.sidebar.selectbox('', speaker_values, index=0)
        
st.sidebar.subheader('Speed')
speed_values = cfg["conf_values"]["speed"]
speed_default = cfg["conf_default"]["speed"]
speed = st.sidebar.selectbox('', speed_values, index=speed_values.index(speed_default))

st.sidebar.subheader('Sampling rate')
sampling_rate_values = cfg["conf_values"]["sampling_rate"]
sampling_rate_default = cfg["conf_default"]["sampling_rate"]
sampling_rate = st.sidebar.selectbox('', sampling_rate_values, index=sampling_rate_values.index(sampling_rate_default))


default_value_text_area = cfg['conf_st_app']['default_value_text_area']
max_height_text_area = None #cfg['conf_st_app']['max_height_text_area']
default_audio_path = cfg['conf_st_app']['default_audio_path']
output_audio_path = cfg['conf_st_app']['output_audio_path']
VALID_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Imk2bEdrM0ZaenhSY1ViMkMzbkVRN3N5SEpsWSIsImtpZCI6Imk2bEdrM0ZaenhSY1ViMkMzbkVRN3N5SEpsWSJ9.eyJhdWQiOiJlZjFkYTlkNC1mZjc3LTRjM2UtYTAwNS04NDBjM2Y4MzA3NDUiLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC9mYTE1ZDY5Mi1lOWM3LTQ0NjAtYTc0My0yOWYyOTUyMjIyOS8iLCJpYXQiOjE1MzcyMzMxMDYsIm5iZiI6MTUzNzIzMzEwNiwiZXhwIjoxNTM3MjM3MDA2LCJhY3IiOiIxIiwiYWlvIjoiQVhRQWkvOElBQUFBRm0rRS9RVEcrZ0ZuVnhMaldkdzhLKzYxQUdyU091TU1GNmViYU1qN1hPM0libUQzZkdtck95RCtOdlp5R24yVmFUL2tES1h3NE1JaHJnR1ZxNkJuOHdMWG9UMUxrSVorRnpRVmtKUFBMUU9WNEtjWHFTbENWUERTL0RpQ0RnRTIyMlRJbU12V05hRU1hVU9Uc0lHdlRRPT0iLCJhbXIiOlsid2lhIl0sImFwcGlkIjoiNzVkYmU3N2YtMTBhMy00ZTU5LTg1ZmQtOGMxMjc1NDRmMTdjIiwiYXBwaWRhY3IiOiIwIiwiZW1haWwiOiJBYmVMaUBtaWNyb3NvZnQuY29tIiwiZmFtaWx5X25hbWUiOiJMaW5jb2xuIiwiZ2l2ZW5fbmFtZSI6IkFiZSAoTVNGVCkiLCJpZHAiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC83MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMjIyNDcvIiwiaXBhZGRyIjoiMjIyLjIyMi4yMjIuMjIiLCJuYW1lIjoiYWJlbGkiLCJvaWQiOiIwMjIyM2I2Yi1hYTFkLTQyZDQtOWVjMC0xYjJiYjkxOTQ0MzgiLCJyaCI6IkkiLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzdWIiOiJsM19yb0lTUVUyMjJiVUxTOXlpMmswWHBxcE9pTXo1SDNaQUNvMUdlWEEiLCJ0aWQiOiJmYTE1ZDY5Mi1lOWM3LTQ0NjAtYTc0My0yOWYyOTU2ZmQ0MjkiLCJ1bmlxdWVfbmFtZSI6ImFiZWxpQG1pY3Jvc29mdC5jb20iLCJ1dGkiOiJGVnNHeFlYSTMwLVR1aWt1dVVvRkFBIiwidmVyIjoiMS4wIn0.D3H6pMUtQnoJAGq6AHd'

# Content
st.title("Text to speech")

sentence = st.text_area(label='Input your text here:',
                        value=default_value_text_area,
                        height=max_height_text_area)
        
if st.button("Generate"):
    data = {'text': sentence}
    print("\n------------------------------------")
    print(f"POST {data}")
    data1 = {'email': 'admin@vlsp.com.vn', 'password': 'admin'}
    header = {"Content-type": 'application/json', "access_token": VALID_TOKEN}

    is_reponse = False
    try:
        response = post('http://localhost:5000/tts', data=json.dumps(data), headers=header)
        is_reponse = True
        print("_____Response____")
    except:
        print("_____No response____")

    if is_reponse:
        #try:
          # write array to file:
        #write(output_audio_path, response["sr"], np.array(response["data"], np.int16))
        #st.audio(output_audio_path, format='audio/wav')

       # except:
        print("Cannot write audio file!!!")
    else:
        st.text('No response from server.')
        
