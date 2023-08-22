# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from load_data import load_data, num_to_char
from load_model import load_model


# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('Logo.png')
    st.title('LipRead')
    st.info('This application is originally developed to help patients with hearing and speaking dis-ability.')

st.title('LipRead') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open(file_path, 'rb') 
        video_bytes = video.read() 
        #st.video(file_path)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        #imageio.mimsave('animation.gif', video, duration=30)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.success(converted_prediction)


        