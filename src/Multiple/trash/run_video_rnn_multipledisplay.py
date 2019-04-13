import IPython
import pandas as pd
import keras
import itertools
import argparse
import logging
import time
import os
import common
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Activation
from keras.layers import LSTM
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
cnt = [0,0,0,0,0,0,0,0]
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
storage = [[],[],[],[],[],[],[],[]]
storage2 = [[],[],[],[],[],[],[],[]]
fps_time = 0
previous_activity = 'null'

activity = ['walking','running','waving','standing']

def samples_to_3D_array(_vector_dim, _vectors_per_sample, _X):
    X_len = len(_X)
    result_array = []
    for sample in range (0,X_len): #should be the 311 samples?
        sample_array = []
        for vector_idx in range (0, _vectors_per_sample):
            start = vector_idx * _vector_dim
            end = start + _vector_dim
            sample_array.append(_X[sample][start:end])
        result_array.append(sample_array)

    return np.asarray(result_array)

X_vector_dim = 36
y_vector_dim = 5
input_shape=(5,36)
timesteps = 5
batch_size = 32
_dropout = 0.1
_activation='relu'
_optimizer='Adam'
model_weights_path = "src/model/pose_model.h5"

print ("Build Keras Timedistributed-LSTM Model...")
print(X_vector_dim)
model = Sequential()
model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
model.add(Dropout(_dropout))
model.add(TimeDistributed(Dense(X_vector_dim*2, activation=_activation))) #(5, 80)
model.add(Dropout(_dropout))
model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation))) #(5, 40)
model.add(Dropout(_dropout))
model.add(TimeDistributed(Dense(int(X_vector_dim/2), activation=_activation))) #(5, 20)
model.add(Dropout(_dropout))
model.add(TimeDistributed(Dense(int(X_vector_dim/4), activation=_activation))) #(5, 10)
model.add(Dropout(_dropout))
model.add(LSTM(int(X_vector_dim/4), dropout=_dropout, recurrent_dropout=_dropout))
model.add(Dense(y_vector_dim,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
model.summary()
model.load_weights(model_weights_path)

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--camera', type=int, default=2)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(0)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        #logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]


        humans = e.inference(image)


        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)


        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break


        if len(humans) is not 0:
            for j in range(len(humans)):
                for i in range(common.CocoPart.Background.value):

                    if i in humans[j].body_parts.keys():
                        storage[j].append(humans[j].body_parts[i].x)
                        storage[j].append(humans[j].body_parts[i].y)
                        if i is 0:
                            xcoor = humans[j].body_parts[i].x * cam.get(3) # width of the video feed
                            ycoor = humans[j].body_parts[i].y * cam.get(4) # height

                        cnt[j] = cnt[j] + 2
                    else:
                        storage[j].append(0)
                        storage[j].append(0)
                        cnt[j] = cnt[j] + 2

                    if cnt[j]%36 is 0:
                        storage2[j].append(storage[j])
                        storage[j]=[]

                    if cnt[j]%180 is 0:
                        X = np.array(storage2[j])
                        test_sample = X
                        test_sample_y_hat = model.predict(test_sample.reshape(1, test_sample.shape[0], test_sample.shape[1]))
                        test_sample_y_hat.shape
                        cv2.putText(image, activity[np.argmax(test_sample_y_hat, axis=1)[0] - 1], (int(xcoor), int(ycoor+10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        previous_activity = activity[np.argmax(test_sample_y_hat, axis=1)[0] - 1]
                        storage2[j] = []

                    else:
                        cv2.putText(image, "    ", (int(xcoor), int(ycoor + 10)), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 2)

            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                        break



    cv2.destroyAllWindows()
