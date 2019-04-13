





































 
                  
                            print("Left Person's Predicted Action:","Backswing1")                    
                            print("Left Person's Predicted Action:","Backswing2")
                            print("Left Person's Predicted Action:","Finish")       
                            print("Left Person's Predicted Action:","Standing")                 
                            print("Right Person's Predicted Action:","Backswing1")                    
                            print("Right Person's Predicted Action:","Backswing2")
                            print("Right Person's Predicted Action:","Finish")       
                            print("Right Person's Predicted Action:","Standing")                 
                        help='for debug purpose, if enabled, speed for inference is dropped.')
                        if np.argmax(test_sample_y0_hat, axis=1)[0] == 1:
                        if np.argmax(test_sample_y0_hat, axis=1)[0] == 2:
                        if np.argmax(test_sample_y0_hat, axis=1)[0] == 3:
                        if np.argmax(test_sample_y0_hat, axis=1)[0] == 4:
                        if np.argmax(test_sample_y1_hat, axis=1)[0] == 1:
                        if np.argmax(test_sample_y1_hat, axis=1)[0] == 2:
                        if np.argmax(test_sample_y1_hat, axis=1)[0] == 3:
                        if np.argmax(test_sample_y1_hat, axis=1)[0] == 4:
                        outfile.write('\n')
                        outfile.write('\n')
                        outfile.write('\n')
                        outfile.write('Prediction ' + str(int(cnt/360)) + ' Finished')
                        outfile.write('This frame is finished')
                        print('\n')
                        record0[i*2+1] = true_humans0.body_parts[i].y 
                        record0[i*2] = true_humans0.body_parts[i].x
                        record1[i*2+1] = true_humans1.body_parts[i].y
                        record1[i*2] = true_humans1.body_parts[i].x
                        storage0.append(record0[i*2+1])
                        storage0.append(record0[i*2])
                        storage0.append(true_humans0.body_parts[i].x)
                        storage0.append(true_humans0.body_parts[i].y)
                        storage1.append(record1[i*2+1])
                        storage1.append(record1[i*2])
                        storage1.append(true_humans1.body_parts[i].x)
                        storage1.append(true_humans1.body_parts[i].y)
                        test_sample0 = X0
                        test_sample1 = X1
                        test_sample_y0_hat = model.predict(test_sample0.reshape(1, test_sample0.shape[0], test_sample0.shape[1]))
                        test_sample_y1_hat = model.predict(test_sample1.reshape(1, test_sample1.shape[0], test_sample1.shape[1]))
                        true_humans0 = humans[0]
                        true_humans0 = humans[1]
                        true_humans1 = humans[0]
                        true_humans1 = humans[1]
                        whole_storage0 = []
                        whole_storage0.append(storage0)
                        whole_storage1 = []
                        whole_storage1.append(storage1)
                        X0 = np.array(whole_storage0)
                        X1 = np.array(whole_storage1)
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (0, 255, 0), 2)
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    cnt = cnt + 2 
                    else:
                    else:
                    else:
                    if cnt%36 is 0:
                    if cnt%360 is 0:
                    if dis_0_0 < dis_0_1:
                    if i in true_humans0.body_parts.keys():
                    if i in true_humans1.body_parts.keys():
                    outfile.write('\n')
                    true_humans0 = humans[0]
                    true_humans0 = humans[1]
                    true_humans1 = humans[0]
                    true_humans1 = humans[1]
                cv2.imshow('tf-pose-estimation result', image)
                cv2.putText(image,"Label = 0",(int(true_humans0.body_parts[0].x * width ), int(true_humans0.body_parts[0].y * height)),  cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)            
                cv2.putText(image,"Label = 1",(int(true_humans1.body_parts[0].x * width ), int(true_humans1.body_parts[0].y * height)),  cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)   
                dis_0_0 = cal_dis(new_head0, pre_head0)
                dis_0_1 = cal_dis(new_head0, pre_head1)
                dis_1_0 = cal_dis(new_head1, pre_head0)
                dis_1_1 = cal_dis(new_head1, pre_head1)
                elif dis_0_0 > dis_0_1 and dis_1_1 > dis_1_0:
                else:
                for i in range(common.CocoPart.Background.value):
                if dis_0_0 < dis_0_1 and dis_1_1 < dis_1_0:
                new_head0 = [humans[0].body_parts[0].x, humans[0].body_parts[0].y]
                new_head1 = [humans[1].body_parts[0].x, humans[1].body_parts[0].y] 
                pass
            break
            canvas = np.zeros_like(image)
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            end = start + _vector_dim
            except KeyError:
            image = canvas
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            sample_array.append(_X[sample][start:end])
            start = vector_idx * _vector_dim
            try:
        #cv2.imshow('tf-pose-estimation result', image)
        #logger.debug('image process+')
        #logger.debug('postprocess+')
        #logger.debug('show+')
        cv2.putText(image,
        elif args.zoom > 1.0:
        for vector_idx in range (0, _vectors_per_sample):
        fps_time = time.time()
        humans = e.inference(image)
        if args.zoom < 1.0:
        if cv2.waitKey(1) == 27:
        if len(humans) is not 0:
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        pre_head0 = [record0[0], record0[1]]
        pre_head1 = [record1[0], record1[1]]
        result_array.append(sample_array)
        ret_val, image = cam.read()
        sample_array = []
        storage0 = []
        storage1 = []
    args = parser.parse_args()
    cam = cv2.VideoCapture(args.video)
    cv2.destroyAllWindows()
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    for sample in range (0,X_len): #should be the 311 samples?
    height = cam.get(4)
    logger.debug('cam read+')
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    out=np.square(coor1[0]-coor2[0])+np.square(coor1[1]-coor2[1])
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=str, default='cmu_640x480', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    result_array = []
    ret_val, image = cam.read()
    return ls2, ls1
    return np.asarray(result_array)
    return np.sqrt(out)
    w, h = model_wh(args.model)
    while True:
    width = cam.get(3)
    X_len = len(_X)
_activation='relu'
_dropout = 0.1
_optimizer='Adam'
batch_size = 32
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
cnt = 0
def cal_dis(coor1, coor2):
def samples_to_3D_array(_vector_dim, _vectors_per_sample, _X):
def swap(ls1, ls2):
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
fps_time = 0
from estimator import TfPoseEstimator
from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Activation
from keras.layers import LSTM
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from networks import get_graph_path, model_wh
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
import argparse
import common
import cv2
import IPython
import itertools
import keras
import logging
import numpy as np
import os
import pandas as pd
import time
input_shape=(10,36)
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
model = Sequential()
model.add(Dense(y_vector_dim,activation='softmax'))
model.add(LSTM(int(X_vector_dim/4), dropout=_dropout, recurrent_dropout=_dropout))
model.add(TimeDistributed(Dense(int(X_vector_dim/2), activation=_activation))) #(5, 20)
model.add(TimeDistributed(Dense(int(X_vector_dim/4), activation=_activation))) #(5, 10)
model.add(TimeDistributed(Dense(X_vector_dim*2, activation=_activation))) #(5, 80)
model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation))) #(5, 40)
model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
model.compile(loss='categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
model.load_weights(model_weights_path)
model.summary()
model_weights_path = "RealTime/model/0213_drivingAllRight/golf_model.h5"
outfile = open('result.txt', 'w')
print ("Build Keras Timedistributed-LSTM Model...")
print(X_vector_dim)
record0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
record1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
storage = []
storage2 = []
whole_storage0 = []
whole_storage1 = [] 
X_vector_dim = 36
y_vector_dim = 5ip