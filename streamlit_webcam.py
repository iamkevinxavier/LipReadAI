#""" Importing the required Dependencies """
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from load_data import load_data, num_to_char
from load_model import load_model

import cv2
import numpy as np
import threading
import time

#""" Set up OpenCV video capture """
video_capture = cv2.VideoCapture(0)
#video_capture.set(3, 640)  # Width
#video_capture.set(4, 480)  # Height
video_capture.set(3, 360)  # Width
video_capture.set(4, 288)  # Height

#""" Global variables """
recording = False
recorded_frames = []
recorded_frames_for_mpg = []
recorded_frames_for_mp4 = []

#""" function to capture the webcam video """
def capture_video():
    global recording, recorded_frames, recorded_frames_for_mpg, recorded_frames_for_mp4
    recorded_frames = []
    recorded_frames_for_mpg = []
    recorded_frames_for_mp4 = []
    recording = True


    fourcc_mpg = cv2.VideoWriter_fourcc(*'MPG1')
    out_mpg = cv2.VideoWriter('captured_video.mpg', fourcc_mpg, 20.0, (360, 288))
    
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'XVID')
    out_mp4 = cv2.VideoWriter('captured_video.mp4', fourcc_mp4, 20.0, (360, 288))

    start_time = time.time()
    while recording:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        recorded_frames.append(frame)
        recorded_frames_for_mpg.append(frame)
        recorded_frames_for_mp4.append(frame)
        
        out_mpg.write(frame)
        out_mp4.write(frame)
        
        elapsed_time = time.time() - start_time
        if elapsed_time >= 30:
            break
    
    recording = False
    out_mpg.release()
    out_mp4.release()



# Set the layout to the streamlit app as wide 
#st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('Logo.png')
    st.title('LipRead')
    st.info('This application is originally developed to help patient who are struggling to communicate.')

st.title('LipRead') 

if st.button("Start Recording"):
    if not recording:
        video_thread = threading.Thread(target=capture_video)
        video_thread.start()
        st.write("Recording...")

if st.button("Stop Recording"):
    if recording:
        recording = False
        st.write("Stopped recording.")
        video_thread.join()
        st.write(f"Captured {len(recorded_frames)} frames.")
        st.video(np.array(recorded_frames))


#""" Variables """
mpg_video = np.array(recorded_frames_for_mpg)
mp4_video = np.array(recorded_frames_for_mp4)

#""" Path """
mpg_path = '..\\data\\webcam_mpg\\*.mpg'
mp4_path = '..\\data\\webcam_mp4\\*.mp4'

#""" Saving it on a specific path """
np.save(mpg_path,mpg_video)
np.save(mp4_path,mp4_video)

#""" Generating 2 columns """
col1, col2 = st.columns(2)

with col1:
    st.info('The video below displays the converted video in mp4 format')
    file_path = '..\\data\\webcam_mp4\\*.mp4'
    video = open(file_path, 'rb') 
    video_bytes = video.read() 
    st.video(video_bytes)

with col2:
    st.info('This is all the machine learning model sees when making a prediction')
    video, annotations = load_data(tf.convert_to_tensor(file_path))
    imageio.mimsave('animation.gif', video, fps=10)
    st.image('animation.gif', width=400) 

    st.info('This is the output of the machine learning model as tokens')
    model = load_model()
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    st.text(decoder)  

    # Convert prediction to text
    st.info('Decode the raw tokens into words')
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
    st.text(converted_prediction)

