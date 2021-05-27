import streamlit as st
import SessionState

from scipy.io.wavfile import write

from init_api import load_model, inference, load_conf


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

# Content
st.title("Text to speech")

sentence = st.text_area(label='Input your text here:',
                        value=default_value_text_area,
                        height=max_height_text_area) 
        


model_text2mel, model_mel2audio, denoiser = load_model(cfg)
session_state = SessionState.get(name='', _audio=None)

if st.button("Generate"):
    data = inference(model_text2mel, model_mel2audio, denoiser, sentence, accent, speed, sampling_rate)
    # write array to file:
    write(output_audio_path, sampling_rate, data)
    print("Writing to " + output_audio_path)
    session_state._audio = True
    st.audio(output_audio_path, format='audio/wav')
else:
    if session_state._audio:
        st.audio(output_audio_path, format='audio/wav')
    else:
        st.audio(default_audio_path, format='audio/wav')
