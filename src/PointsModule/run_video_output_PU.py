import argparse
import logging
import time
import os
import common
import cv2
import numpy as np
import math

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
storage=[]
cnt = 0
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)   #以上为输出控制台的标准格式

fps_time = 0

def GetAngle(coor1,coor2,coor3):
    L1=CalDis(coor2,coor3)
    L2=CalDis(coor1,coor3)
    L3=CalDis(coor1,coor2)
    Angle=CalAngle(L1,L2,L3)
    return Angle

def CalDis(coor1,coor2):
    out=np.square(coor1[0]-coor2[0])+np.square(coor1[1]-coor2[1])
    return np.sqrt(out)

def CalAngle(L1,L2,L3):
    out=(np.square(L2)+np.square(L3)-np.square(L1))/(2*L2*L3)
    return math.acos(out)*(180/math.pi)

def CheckArm(coor1,coor2,coor3):
    angle=GetAngle(coor1,coor2,coor3)
    if angle>170:
        return 1
    elif angle<140:
        return 2
    else:
        return 0

def CheckTurn(LeftAn,RightAn,LeftSh,RightSh):
    LenShoulder = LeftSh[0] - RightSh[0]
    LenAnkle = LeftAn[0] - RightAn[0]
    if LenShoulder<0.4*LenAnkle:
        return True
    else:
        return False

def CheckHand(LHand,RHand,RElbow):
    HandDis=CalDis(LHand,RHand)
    ArmDis=CalDis(RHand,RElbow)
    if HandDis<0.6*ArmDis:
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='cmu_640x480', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()    #读取命令行中的指令，--指的是附加的
    file = open(args.output, "a+")
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.video)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    LCurve=0
    LStraight=0
    RCurve=0
    RStraight=0
    TooW=0
    TooN=0
    Slen=0
    WrongH=0
    RightH=0

    while True:
        ret_val, image = cam.read()

        #logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)  #返回与image具有同样形状的数组
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
                    (0, 255, 0), 2) #将FPS的数据放上视频中
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')
        #print(len(humans.body_parts))  #Display the no of joints detected
        LShoulder = []
        LWrist = []
        LElbow=[]
        RHip =[]
        for i in range(common.CocoPart.Background.value):
            if i in humans[0].body_parts.keys():
                if i in [5]:
                    LShoulder.append(humans[0].body_parts[i].x)
                    LShoulder.append(humans[0].body_parts[i].y)
                if i in [6]:
                    LElbow.append(humans[0].body_parts[i].x)
                    LElbow.append(humans[0].body_parts[i].y)
                if i in [7]:
                    LWrist.append(humans[0].body_parts[i].x)
                    LWrist.append(humans[0].body_parts[i].y)
                if i in [8]:
                    RHip.append(humans[0].body_parts[i].x)
                    RHip.append(humans[0].body_parts[i].y)
                cnt = cnt + 2       
            else:
                cnt = cnt + 2
                
            if cnt%26 is 0:
                if len(LElbow)==2 and len(LShoulder)==2 and len(LWrist)==2:
                    LAngle = GetAngle(LElbow, LShoulder, LWrist)
                    print("The left arm's angle: "+str(LAngle))
                    file.write("The left arm's angle: "+str(LAngle)+'\n')
                    if CheckArm(LElbow,LWrist,LShoulder)== 1:
                        print("Left Arm is too straight")
                    elif CheckArm(LElbow,LWrist,LShoulder)== 2:
                        print("Left Arm is too curve")
                    else:
                        print("Your left arm is right")


                if len(RHip)==2 and len(LWrist)==2:
                    print("The height of hip: "+str(RHip[0]))
                    print("The height of left hand: "+str(LWrist[0]))
                    file.write("The height of nose: "+str(RHip[0])+'\n')
                    file.write("The height of left hand: "+str(LWrist[0])+'\n')
                    if  LWrist[0]>RHip[0]:
                        print("Your waving position is right")
                    else:
                        print("You aer waving too high")
                file.write('\n'+'\n'+'\n')

    file.close()
    cv2.destroyAllWindows()
