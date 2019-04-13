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
input_shape=(10,36)
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
outfile = open('result.txt', 'w')

model.load_weights(model_weights_path)


record = [[],[],[],[]]
record[0] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
record[1] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

whole_storage = [[],[]]
true_humans = [[],[]]
storage = [[],[]]
X = [[],[]]
test_sample = [[],[]]
test_sample_y_hat = [[],[]]
cnt = [0,0]

def cal_dis(coor1, coor2):
    out=np.square(coor1[0]-coor2[0])+np.square(coor1[1]-coor2[1])
    return np.sqrt(out)

def swap(ls1, ls2):
    return ls2, ls1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='cmu_640x480', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    width = cam.get(3)
    height = cam.get(4)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

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

        #logger.debug('image process+')
        humans = e.inference(image)

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        #cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

        pre_head0 = [record[0][0], record[0][1]]
        pre_head1 = [record[1][0], record[1][1]]

        storage[0] = []
        storage[1] = []

        if len(humans) is not 0:

            try:zx

                if len(humans) == 1:
                    new_head = [humans[0].body_parts[0].x, humans[0].body_parts[0].y]
                    dis_0 = cal_dis(new_head, pre_head0)
                    dis_1 = cal_dis(new_head, pre_head1)
                    if dis_0 > dis_1:
                        true_label = 1
                    else:
                        true_label = 0

                    cv2.putText(image,"Label{}".format(true_label),(int(humans[0].body_parts[0].x * width ), int(humans[0].body_parts[0].y * height)),  cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)

                    for i in range(common.CocoPart.Background.value):
                        if i in humans[0].body_parts.keys():
                            storage[true_label].append(humans[0].body_parts[i].x)
                            storage[true_label].append(humans[0].body_parts[i].y)
                            record[true_label][i*2] = humans[0].body_parts[i].x
                            record[true_label][i*2+1] = humans[0].body_parts[i].y
                        else:
                            storage[true_label].append(record[true_label][i*2])
                            storage[true_label].append(record[true_label][i*2+1])

                        cnt[true_label] += 2

                        if cnt[true_label]%36 is 0:
                            whole_storage[true_label].append(storage[true_label])

                        if cnt[true_label]%360 is 0:
                            X[true_label] = np.array(whole_storage[true_label])
                            test_sample[true_label] = X[true_label]
                            test_sample_y_hat[true_label] = model.predict(test_sample[true_label].reshape(1, test_sample[true_label].shape[0], test_sample[true_label].shape[1]))
                            if np.argmax(test_sample_y_hat[true_label], axis=1)[0] == 1:
                                print("Person{0}'s Predicted Action:  Finish".format(true_label))
                            if np.argmax(test_sample_y_hat[true_label], axis=1)[0] == 2:
                                print("Person{0}'s Predicted Action:  Standing".format(true_labelj))
                            if np.argmax(test_sample_y_hat[jtrue_label], axis=1)[0] == 3:
                                print("Person{0}'s Predicted Action:  Backswing1".format(true_label))
                            if np.argmax(test_sample_y_hat[true_label], axis=1)[0] == 4:
                                print("Person{0}'s Predicted Action:  Backswing2".format(true_label))

                            whole_storage[true_label] = []


                elif len(humans) == 2:

                    new_head0 = [humans[0].body_parts[0].x, humans[0].body_parts[0].y]
                    new_head1 = [humans[1].body_parts[0].x, humans[1].body_parts[0].y]

                    dis_0_0 = cal_dis(new_head0, pre_head0)
                    dis_0_1 = cal_dis(new_head0, pre_head1)
                    dis_1_1 = cal_dis(new_head1, pre_head1)
                    dis_1_0 = cal_dis(new_head1, pre_head0)

                    if dis_0_0 < dis_0_1 and dis_1_1 < dis_1_0:
                        true_humans[0] = humans[0]
                        true_humans[1] = humans[1]

                    elif dis_0_0 > dis_0_1 and dis_1_1 > dis_1_0:
                        true_humans[0] = humans[1]
                        true_humans[1] = humans[0]

                    else:
                        if dis_0_0 < dis_0_1:
                            true_humans[0] = humans[0]
                            true_humans[1] = humans[1]
                        else:
                            true_humans[0] = humans[1]
                            true_humans[1] = humans[0]

                    cv2.putText(image,"Label0",(int(true_humans[0].body_parts[0].x * width ), int(true_humans[0].body_parts[0].y * height)),  cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)
                    cv2.putText(image,"Label1",(int(true_humans[1].body_parts[0].x * width ), int(true_humans[1].body_parts[0].y * height)),  cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)


                    for i in range(common.CocoPart.Background.value):
                        for j in range(len(humans)):
                            if i in true_humans[j].body_parts.keys():
                                storage[j].append(true_humans[j].body_parts[i].x)
                                storage[j].append(true_humans[j].body_parts[i].y)
                                record[j][i*2] = true_humans[j].body_parts[i].x
                                record[j][i*2+1] = true_humans[j].body_parts[i].y
                            else:
                                storage[j].append(record[j][i*2])
                                storage[j].append(record[j][i*2+1])

                            cnt[j] = cnt[j] + 2
                            outfile.write('\n')

                            if cnt[j]%36 is 0:
                                whole_storage[j].append(storage[j])
                                outfile.write('This frame is finished')
                                outfile.write('\n')

                            if cnt[j]%360 is 0:
                                X[j] = np.array(whole_storage[j])
                                test_sample[j] = X[j]
                                test_sample_y_hat[j] = model.predict(test_sample[j].reshape(1, test_sample[j].shape[0], test_sample[j].shape[1]))
                                if np.argmax(test_sample_y_hat[j], axis=1)[0] == 1:
                                    print("Person{0}'s Predicted Action:  Finish".format(j))
                                if np.argmax(test_sample_y_hat[j], axis=1)[0] == 2:
                                    print("Person{0}'s Predicted Action:  Standing".format(j))
                                if np.argmax(test_sample_y_hat[j], axis=1)[0] == 3:
                                    print("Person{0}'s Predicted Action:  Backswing1".format(j))
                                if np.argmax(test_sample_y_hat[j], axis=1)[0] == 4:
                                    print("Person{0}'s Predicted Action:  Backswing2".format(j))

                                whole_storage[j] = []
                                outfile.write('\n')
                                outfile.write('\n')
                                print('\n')

                else:          # len(humans) > 2, ignore this frame
                    pass

            except KeyError:      # if the head is not detected, ignore this frame
                pass


        cv2.imshow('tf-pose-estimation result', image)

    cv2.destroyAllWindows()
