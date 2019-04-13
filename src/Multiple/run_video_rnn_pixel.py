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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras.models import load_model
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

cnt = 0
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
storage = []
storage2 = []
fps_time = 0

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
batch_size = 32
_dropout = 0.1
_activation='relu'
_optimizer='Adam'
model_weights_path = "RealTime/model/0213_drivingAllRight/golf_model.h5"

print ("Build Keras Timedistributed-LSTM Model...")
print(X_vector_dim)
model = Sequential()
model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
model.add(TimeDistributed(Dense(X_vector_dim*2, activation=_activation))) #(5, 80)
model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation))) #(5, 40)
model.add(TimeDistributed(Dense(int(X_vector_dim/2), activation=_activation))) #(5, 20)
model.add(TimeDistributed(Dense(int(X_vector_dim/4), activation=_activation))) #(5, 10)
model.add(LSTM(int(X_vector_dim/4), dropout=_dropout, recurrent_dropout=_dropout))
model.add(Dense(y_vector_dim,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
model.summary()

model.load_weights(model_weights_path)

record0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
record1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def cal_dis(coor1, coor2):
    out=np.square(coor1[0]-coor2[0])+np.square(coor1[1]-coor2[1])
    return np.sqrt(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='cmu_640x480', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.video)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    whole_storage0 = []
    whole_storage1 = []

    while True:
        ret_val, image = cam.read()

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

        print(len(image))
        print(image[0])
        print(len(image[0]))

        #logger.debug('image process+')
        # humans = e.inference(image)

        # #logger.debug('postprocess+')
        # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # #logger.debug('show+')
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        # cv2.imshow('tf-pose-estimation result', image)
        # fps_time = time.time()
        # if cv2.waitKey(1) == 27:
        #     break

        # pre_head0 = [record0[0], record0[1]]
        # pre_head1 = [record1[0], record1[1]]

        # storage0 = []
        # storage1 = []

        # if len(humans) is not 0:
        #     for i in range(common.CocoPart.Background.value):
        #         if i in humans[0].body_parts.keys():
        #             storage0.append(humans[0].body_parts[i].x)
        #             storage0.append(humans[0].body_parts[i].y)
        #             record0[i*2] = humans[0].body_parts[i].x
        #             record0[i*2+1] = humans[0].body_parts[i].y
        #             #3、修改成没有读到点就取上一个点的值 
        #         else:
        #             storage0.append(record0[i*2])
        #             storage0.append(record0[i*2+1])

        #         if i in humans[1].body_parts.keys():
        #             storage1.append(humans[1].body_parts[i].x)
        #             storage1.append(humans[1].body_parts[i].y)
        #             record1[i*2] = humans[1].body_parts[i].x
        #             record1[i*2+1] = humans[1].body_parts[i].y
        #         else:
        #             storage1.append(record0[i*2])
        #             storage1.append(record0[i*2+1])
              
        #         cnt = cnt + 2 


        #         if cnt%36 is 0:

        #             new_head0 = [storage0[0], storage0[1]]
        #             new_head1 = [storage1[0], storage1[1]] 

        #             dis_0_0 = cal_dis(new_head0, pre_head0)
        #             dis_0_1 = cal_dis(new_head0, pre_head1)
        #             dis_1_1 = cal_dis(new_head1, pre_head1)
        #             dis_1_0 = cal_dis(new_head1, pre_head0)

        #             if dis_0_0 > dis_0_1 and dis_1_1 > dis_1_0:
        #                 whole_storage0.append(storage0)
        #                 whole_storage1.append(storage1)

        #             elif dis_0_0 < dis_0_1 and dis_1_1 < dis_1_0:
        #                 whole_storage0.append(storage1)
        #                 whole_storage1.append(storage0)

        #             else:
        #                 if dis_0_0 > dis_0_1:
        #                     whole_storage0.append(storage0)
        #                     whole_storage1.append(storage1)
        #                 else:
        #                     whole_storage0.append(storage1)
        #                     whole_storage1.append(storage0)    


        #         if cnt%180 is 0:
        #             X0 = np.array(whole_storage0)
        #             test_sample0 = X0
        #             test_sample_y0_hat = model.predict(test_sample0.reshape(1, test_sample0.shape[0], test_sample0.shape[1]))
        #             if np.argmax(test_sample_y0_hat, axis=1)[0] == 1:
        #                 print("Right Person's Predicted Action:","Finish")       
        #             if np.argmax(test_sample_y0_hat, axis=1)[0] == 2:
        #                 print("Right Person's Predicted Action:","Standing")                 
        #             if np.argmax(test_sample_y0_hat, axis=1)[0] == 3:
        #                 print("Right Person's Predicted Action:","Backswing1")                    
        #             if np.argmax(test_sample_y0_hat, axis=1)[0] == 4:
        #                 print("Right Person's Predicted Action:","Backswing2")

        #             X1 = np.array(whole_storage1)
        #             test_sample1 = X1
        #             test_sample_y1_hat = model.predict(test_sample1.reshape(1, test_sample1.shape[0], test_sample1.shape[1]))
        #             if np.argmax(test_sample_y1_hat, axis=1)[0] == 1:
        #                 print("Left Person's Predicted Action:","Finish")       
        #             if np.argmax(test_sample_y1_hat, axis=1)[0] == 2:
        #                 print("Left Person's Predicted Action:","Standing")                 
        #             if np.argmax(test_sample_y1_hat, axis=1)[0] == 3:
        #                 print("Left Person's Predicted Action:","Backswing1")                    
        #             if np.argmax(test_sample_y1_hat, axis=1)[0] == 4:
        #                 print("Left Person's Predicted Action:","Backswing2")

        #             print('\n')
        #             whole_storage0 = []
        #             whole_storage1 = []

    cv2.destroyAllWindows()
