import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


from tensorflow.keras.models import load_model


mediapipe_holistic = mp.solutions.holistic
mediapipe_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('MP_Data')
list_of_gest = ['peace', 'like', 'dislike', 'ok']
actions = np.array(list_of_gest)
no_sequences = 20
sequence_length = 20


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):

    mediapipe_drawing.draw_landmarks(image, results.left_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS)
    mediapipe_drawing.draw_landmarks(image, results.right_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):

    mediapipe_drawing.draw_landmarks(image, results.left_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS,
                             mediapipe_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mediapipe_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )

    mediapipe_drawing.draw_landmarks(image, results.right_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS,
                             mediapipe_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mediapipe_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


def camera_capture_prediction(status):
    model = load_model('action.h5')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if status == 1:

        cap.release()
        cv2.destroyAllWindows()

    elif status == 2:

        list_of_gest = ['peace', 'like', 'dislike', 'ok']
        actions = np.array(list_of_gest)

        colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 217, 245)]

        num_sequence = []
        list_sentence = []
        list_predictions = []
        threshold = 0.91

        with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                success, frame = cap.read()  # read the camera frame
                if not success:
                    break
                else:

                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)



                    draw_styled_landmarks(image, results)

                    keypoints = extract_keypoints(results)
                    num_sequence.append(keypoints)
                    num_sequence = num_sequence[-no_sequences:]

                    if len(num_sequence) == no_sequences:
                        res = model.predict(np.expand_dims(num_sequence, axis=0))[0]

                        list_predictions.append(np.argmax(res))

                        if np.unique(list_predictions[-no_sequences:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:

                                if len(list_sentence) > 0:
                                    if actions[np.argmax(res)] != list_sentence[-1]:
                                        list_sentence.append(actions[np.argmax(res)])
                                else:
                                    list_sentence.append(actions[np.argmax(res)])

                        if len(list_sentence) > 5:
                            list_sentence = list_sentence[-5:]

                        image = prob_viz(res, actions, image, colors)

                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(list_sentence), (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    ret, buffer = cv2.imencode('.jpg', image)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def label_function(status):

    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if status == 1:
        print('label fun if condition')
        cap.release()
        cv2.destroyAllWindows()

    elif status == 2:
        print('label fun else condition')

        with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                for action in actions:

                    ret, frame4 = cap.read()
                    image, results = mediapipe_detection(frame4, holistic)
                    cv2.putText(image, 'Next Action is {}'.format(action), (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4, cv2.LINE_AA)

                    ret4, buffer4 = cv2.imencode('.jpg', image)
                    frame1 = buffer4.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')

                    cv2.waitKey(5000)

                    for sequence in range(no_sequences):
                        for frame_num in range(sequence_length):

                            ret, frame = cap.read()

                            image, results = mediapipe_detection(frame, holistic)

                            draw_styled_landmarks(image, results)

                            if frame_num == 0:

                                cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                                cv2.putText(image,
                                            'Collecting frames for {} Video {}'.format(action,
                                                                                                         sequence),
                                            (15, 12),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                                ret, buffer1 = cv2.imencode('.jpg', image)
                                frame2 = buffer1.tobytes()

                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

                                cv2.waitKey(2000)
                            else:

                                cv2.putText(image,
                                            'Collecting frames for {} Video Number {}'.format(action,
                                                                                                           sequence),
                                            (15, 12),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                                ret, buffer2 = cv2.imencode('.jpg', image)
                                frame3 = buffer2.tobytes()

                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame3 + b'\r\n')

                            keypoints = extract_keypoints(results)
                            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                            np.save(npy_path, keypoints)

                cap.release()
                cv2.destroyAllWindows()


def training_dataset():
    global wait_message
    wait_message = 'Please wait until you see complete word displayed'

    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(no_sequences, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=1500, callbacks=[tb_callback])

    model.save('action.h5')


def train_dataset_cnn():
    global wait_message
    wait_message = 'Please wait until you see complete word displayed'

    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, strides=1, padding="causal", activation='relu', input_shape=(no_sequences, 126)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 5, strides=1, padding="causal", activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 5, strides=1, padding="causal", activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=1500, callbacks=[tb_callback])

    model.save('action.h5')



def extract_keypoints(results):
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if prob > 1:
            print()
        else:
            rate = prob
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 200), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num] + ' ' + str("%.2f" % rate), (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame
# def prob_viz(res, action, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
#         cv2.putText(output_frame, action[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
#                     cv2.LINE_AA)
#
#     return output_frame
