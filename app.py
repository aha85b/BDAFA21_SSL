from flask import Flask, render_template, Response, Request, jsonify
import cv2
from open_cv import camera_capture_prediction, label_function, training_dataset, train_dataset_cnn
import multiprocessing
import threading
from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)


change_cam_status = 3
change_label_status = 3

DATA_PATH = os.path.join('MP_Data')

no_sequences = 10

sequence_length = 10

start_folder = 10

# List of gestures
list_of_gest = ['hello', 'thanks', 'iloveyou', 'ok']
# Actions that we try to detect
actions = np.array(list_of_gest)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 217, 245)]


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/camera_cap')
def camera_cap():
    isopened = 1

    if change_cam_status == 2:
        isopened = 2


    print('hi camera capture',isopened)
    return Response(camera_capture_prediction(isopened), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return 'hi camera capture return'



@app.route('/open_cam')
def open_cam():

    global change_cam_status

    change_cam_status = 2

    camera_cap()

    print('hi open cam')
    return render_template('predict.html')


@app.route('/close_cam')
def close_cam():
    print('hi close cam')

    global change_cam_status

    change_cam_status = 4

    camera_cap()
    return render_template('predict.html')


@app.route('/camera')
def camera():
    return render_template('predict.html')


@app.route('/label')
def label():
    return render_template('label.html')


@app.route('/label_fun')
def label_fun():
    isOpen = 1
    global change_label_status

    if change_label_status == 2:
        isOpen = 2

    return Response(label_function(isOpen), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/open_label_cam')
def open_label_cam():

    global change_label_status
    change_label_status = 2

    label_fun()

    return render_template('label.html')


@app.route('/close_label_cam')
def close_label_cam():

    global change_label_status
    change_label_status = 1

    label_fun()

    return render_template('label.html')


@app.route('/train_dataset')
def train_dataset():

    training_dataset()
    message_value = ' COMPLETED '
    return render_template('label.html', value=message_value)

@app.route('/train_cnn')
def train_cnn():
    train_dataset_cnn()
    message_value = ' COMPLETED '
    return render_template('label.html', value=message_value)


@app.route('/about_us')
def about_us():
    return render_template('about.html')


if __name__ == '__main__':
    app.run()
