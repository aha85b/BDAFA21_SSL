from flask import Flask, render_template, Response, request, jsonify
import cv2
from open_cv import camera_capture_prediction, label_function, training_dataset, train_dataset_cnn, \
    list_of_predicted_words, extract_keypoints, draw_styled_landmarks, prob_viz, mediapipe_detection
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
import datetime
from playsound import playsound
from gtts import gTTS


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

mediapipe_holistic = mp.solutions.holistic
mediapipe_drawing = mp.solutions.drawing_utils



change_cam_status = 3
change_label_status = 3

DATA_PATH = os.path.join('MP_Data')

no_sequences = 20

sequence_length = 20

start_folder = 10

num_of_sequ = 20

# List of gestures
# list_of_gest = ['hello', 'thanks', 'iloveyou', 'ok']
# # Actions that we try to detect
# actions = np.array(list_of_gest)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 217, 245)]

list_of_names = []
actions = np.array(list_of_names)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('label.html')


@app.route('/camera_cap')
def camera_cap():
    isopened = 1

    if change_cam_status == 2:
        isopened = 2


    print('hi camera capture',isopened)
    return Response(camera_capture_prediction(isopened, list_of_names, num_of_sequ), mimetype='multipart/x-mixed-replace; boundary=frame')
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


@app.route('/camera', methods=['POST', "GET"])
def camera():
    if request.method == "POST":
        user = request.form["name"]
        list_of_names.append(user)
        # print(user)
        return render_template('predict.html', list=list_of_names)
    else:
        word = "I am trying to pass a text in here"
        return render_template('predict.html', list=list_of_names)


@app.route('/label', methods=['POST', 'GET'])
def label():
    global num_of_sequ
    if request.method == "POST":
        user = request.form["word1"]
        list_of_names.append(user)
        user2 = request.form["word2"]
        list_of_names.append(user2)
        user3 = request.form["word3"]
        list_of_names.append(user3)
        user4 = request.form["word4"]
        list_of_names.append(user4)
        squence_num = request.form["sequences"]
        num_of_sequ = squence_num

        # print(user)
        return render_template('label.html', list=list_of_names, sequences=num_of_sequ)
    else:
        word = "I am trying to pass a text in here"
        return render_template('label.html', list=list_of_names, sequences=num_of_sequ)


@app.route('/label_fun')
def label_fun():
    isOpen = 1
    global change_label_status

    if change_label_status == 2:
        isOpen = 2

    print(num_of_sequ)

    return Response(label_function(isOpen, list_of_names, num_of_sequ), mimetype='multipart/x-mixed-replace; boundary=frame')


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

    training_dataset(list_of_names, num_of_sequ)
    message_value = ' COMPLETED '
    return render_template('label.html', value=message_value)

@app.route('/train_cnn')
def train_cnn():
    train_dataset_cnn(list_of_names, num_of_sequ)
    message_value = ' COMPLETED '
    return render_template('label.html', value=message_value)


@app.route('/human_ai')
def human_ai_page():
    return render_template('human_ai.html')


@app.route('/about_us')
def about_us():
    return render_template('about.html')


def list_of_user_gestures():
    if not list_of_names:
        return "list is null"
    else:
        return list_of_names



# def camera_capture_prediction_app(status):
#     model = load_model('action.h5')
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#
#     if status == 1:
#
#         cap.release()
#         cv2.destroyAllWindows()
#
#     elif status == 2:
#
#         # list_of_gest = ['peace', 'like', 'dislike', 'okay']
#         actions = np.array(list_of_names)
#
#         colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 217, 245)]
#
#         num_sequence = []
#         list_sentence = []
#         list_predictions = []
#         threshold = 0.68
#
#         with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             while cap.isOpened():
#                 datetime_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
#                 file_name = "output" + datetime_string + ".mp3"
#                 success, frame = cap.read()  # read the camera frame
#                 if not success:
#                     break
#                 else:
#
#                     ret, frame = cap.read()
#
#                     # Make detections
#                     image, results = mediapipe_detection(frame, holistic)
#
#                     draw_styled_landmarks(image, results)
#
#                     keypoints = extract_keypoints(results)
#                     num_sequence.append(keypoints)
#                     num_sequence = num_sequence[-no_sequences:]
#
#
#
#                     if len(num_sequence) == no_sequences:
#                         res = model.predict(np.expand_dims(num_sequence, axis=0))[0]
#                         list_predictions.append(np.argmax(res))
#
#                         if np.unique(list_predictions[-no_sequences:])[0] == np.argmax(res):
#                             if res[np.argmax(res)] > threshold:
#
#                                 if len(list_sentence) > 0:
#                                     # global file_name
#                                     # datetime_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
#                                     if actions[np.argmax(res)] != list_sentence[-1]:
#                                         list_sentence.append(actions[np.argmax(res)])
#                                         text = "Did you mean to say" + actions[np.argmax(res)]
#                                         output = gTTS(text=text, lang="en", slow=False)
#                                         # datetime_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
#                                         # file_name = "output"+datetime_string+".mp3"
#                                         output.save(file_name)
#                                         print(file_name,"first if")
#                                         playsound(file_name, True)
#                                         os.remove(file_name)
#                                         # val = input('Enter yes or no: ')
#                                         # print(val)
#                                         #playsound("C:/Users/dhh3hb/Documents/GitHub/BDAFA21_SSL/output.mp3")
#
#
#                                 else:
#                                     list_sentence.append(actions[np.argmax(res)])
#                                     image = prob_viz(res, actions, image, colors)
#                                     text = "Did you mean to say" + actions[np.argmax(res)]
#                                     output = gTTS(text=text, lang="en", slow=False)
#                                     # datetime_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
#                                     # file_name = "output" + datetime_string + ".mp3"
#                                     output.save(file_name)
#                                     print(file_name, "first else")
#                                     playsound(file_name, True)
#                                     os.remove(file_name)
#                                     # val = input('Enter yes or no: ')
#                                     # print(val)
#                                     #playsound("C:/Users/dhh3hb/Documents/GitHub/BDAFA21_SSL/output.mp3")
#
#                         if len(list_sentence) > 5:
#                             list_sentence = list_sentence[-5:]
#
#                         image = prob_viz(res, actions, image, colors)
#
#
#
#                     cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
#                     cv2.putText(image, ' '.join(list_sentence), (30, 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#                     ret, buffer = cv2.imencode('.jpg', image)
#                     frame = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#                     cv2.waitKey(20)
#
#
# def label_function_app(status):
#
#     for action in actions:
#         for sequence in range(no_sequences):
#             try:
#                 os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#             except:
#                 pass
#
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#
#     if status == 1:
#         print('label fun if condition')
#         cap.release()
#         cv2.destroyAllWindows()
#
#     elif status == 2:
#         print('label fun else condition')
#
#         with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             while cap.isOpened():
#
#                 for action in actions:
#
#                     ret, frame4 = cap.read()
#                     image, results = mediapipe_detection(frame4, holistic)
#                     cv2.putText(image, 'Next Action is {}'.format(action), (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4, cv2.LINE_AA)
#
#                     ret4, buffer4 = cv2.imencode('.jpg', image)
#                     frame1 = buffer4.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
#
#                     cv2.waitKey(5000)
#
#                     for sequence in range(no_sequences):
#                         for frame_num in range(sequence_length):
#
#                             ret, frame = cap.read()
#
#                             image, results = mediapipe_detection(frame, holistic)
#
#                             draw_styled_landmarks(image, results)
#
#                             if frame_num == 0:
#
#                                 cv2.putText(image, 'STARTING COLLECTION', (120, 200),
#                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                                 cv2.putText(image,
#                                             'Collecting frames for {} Video {}'.format(action,
#                                                                                                          sequence),
#                                             (15, 12),
#                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#
#                                 ret, buffer1 = cv2.imencode('.jpg', image)
#                                 frame2 = buffer1.tobytes()
#
#                                 yield (b'--frame\r\n'
#                                        b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
#
#                                 cv2.waitKey(2000)
#                             else:
#
#                                 cv2.putText(image,
#                                             'Collecting frames for {} Video Number {}'.format(action,
#                                                                                                            sequence),
#                                             (15, 12),
#                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#
#                                 ret, buffer2 = cv2.imencode('.jpg', image)
#                                 frame3 = buffer2.tobytes()
#
#                                 yield (b'--frame\r\n'
#                                        b'Content-Type: image/jpeg\r\n\r\n' + frame3 + b'\r\n')
#
#                             keypoints = extract_keypoints(results)
#                             npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                             np.save(npy_path, keypoints)
#
#                 cap.release()
#                 cv2.destroyAllWindows()
#



if __name__ == '__main__':
    app.run()
