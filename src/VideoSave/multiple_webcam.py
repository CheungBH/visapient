import argparse
import logging
import time
import cv2
import numpy as np
import time
import ctypes

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
recording_time = 200 # seconds
fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368',
                        help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()
    print("Start in 2 seconds")
    time.sleep(2)


    print("Start now:")
    #camera = [cv2.VideoCapture(0), cv2.VideoCapture(1),cv2.VideoCapture(2),cv2.VideoCapture(3),cv2.VideoCapture(4)]
    camera = [cv2.VideoCapture(0), cv2.VideoCapture(1)]

    image = []
    ret_val = []
    ret_val0, image0 = camera[0].read()
    ret_val1, image1 = camera[1].read()
    #ret_val2, image2 = camera[2].read()
    #ret_val3, image3 = camera[3].read()
    #ret_val4, image4 = camera[4].read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out0 = cv2.VideoWriter('00.avi', fourcc, 20.0, (640, 480))
    out1 = cv2.VideoWriter('01.avi', fourcc, 20.0, (640, 480))
    #out2 = cv2.VideoWriter('01.avi', fourcc, 20.0, (640, 480))
    #out3 = cv2.VideoWriter('02.avi', fourcc, 20.0, (640, 480))
    #out4 = cv2.VideoWriter('04.avi', fourcc, 20.0, (640, 480))
    start = time.time()
    player=ctypes.windll.kernel32
    player.Beep(1000,200)
    while True:

        ret_val0, image0 = camera[0].read()
        ret_val1, image1 = camera[1].read()
        #ret_val2, image2 = camera[2].read()
        #ret_val3, image3 = camera[3].read()
		#ret_val4, image4 = camera[4].read()
		
		
        out0.write(image0)
        out1.write(image1)
        #out2.write(image2)
        #out3.write(image3)
        #out4.write(image4)
        cv2.imshow('tf-pose-estimation result0', image0)
        cv2.imshow('tf-pose-estimation result1', image1)
        #cv2.imshow('tf-pose-estimation result2', image2)
        #cv2.imshow('tf-pose-estimation result3', image3)
        #cv2.imshow('tf-pose-estimation result4', image4)


        print(time.time())

        if time.time() >= start + recording_time:
             print("over")
             break

        if cv2.waitKey(1) == 27 :
            break

    player=ctypes.windll.kernel32
    player.Beep(1000,200)
    cv2.destroyAllWindows()
