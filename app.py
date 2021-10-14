import streamlit as st
from scipy.io.wavfile import write
from requests import post
import numpy as np
import yaml

@st.cache
def load_conf():
    with open("config.yml", "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, yaml.Loader)
    return cfg
cfg = load_conf()

# Side bar
st.sidebar.subheader('Accent')
accent_values = cfg["conf_values"]["accent"]
accent_default = cfg["conf_default"]["accent"]
accent = st.sidebar.selectbox('', accent_values, index=accent_values.index(accent_default))
        
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
	@@ -37,20 +40,27 @@
                        value=default_value_text_area,
                        height=max_height_text_area) 

if st.button("Generate"):
    data = {'text': sentence, 'accent': accent, 'speed': speed, 'sr': sampling_rate}
    print("\n------------------------------------")
    print(f"POST {data}")

    is_reponse = False
    try:
        reponse = post('http://localhost:5000/inference', data=data).json()
        is_reponse = True
        print("_____Response____")
    except:
        print("_____No response____")

    if is_reponse:
        try:
            # write array to file:
            write(output_audio_path, reponse["sr"], np.array(reponse["data"], np.int16))
            print("Writing to " + output_audio_path + " successfully")
            st.audio(output_audio_path, format='audio/wav')
        except:
            print("Cannot write audio file!!!")
    else:
        st.text('No response from server.')
