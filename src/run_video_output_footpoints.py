import argparse
import logging
import time
import os
import common
import cv2
import numpy as np

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
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='cmu_640x480', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()
    file = open(args.output, "a+")
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
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

        #logger.debug('image process+')
        humans = e.inference(image)

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')
        #print(len(humans.body_parts))  #Display the no of joints detected
        RAnkle = []
        LAnkle = []
        RShoulder = []
        LShoulder = []

        if len(humans) is not 0:
            for i in range(common.CocoPart.Background.value):

                if i in humans[0].body_parts.keys():
                    # print(humans[0].body_parts.keys())
                    # print(type(humans[0].body_parts.keys()))
                    # print(i,": ",humans[0].body_parts[i].x ," ",humans[0].body_parts[i].y )

                    if i in [10]:
                        RAnkle.append(humans[0].body_parts[i].x)
                        RAnkle.append(humans[0].body_parts[i].y)

                        # print("RAnkle:",humans[0].body_parts[i].x ," ",humans[0].body_parts[i].y)
                        # file.write("RAnkle:")
                        # file.write(str(humans[0].body_parts[i].x))
                        # file.write(" ")
                        # file.write("RX")
                        # file.write(str(humans[0].body_parts[i].y))
                        # file.write(" ")
                        # file.write("RY")

                    if i in [13]:
                        LAnkle.append(humans[0].body_parts[i].x)
                        LAnkle.append(humans[0].body_parts[i].y)

                        # print("LAnkle:",humans[0].body_parts[i].x ," ",humans[0].body_parts[i].y)
                        # file.write("LAnkle:")
                        # file.write(str(humans[0].body_parts[i].x))
                        # file.write(" ")
                        # file.write("LX")
                        # file.write(str(humans[0].body_parts[i].y))
                        # file.write(" ")
                        # file.write("LY")

                    if i in [2]:
                        RShoulder.append(humans[0].body_parts[i].x)
                        RShoulder.append(humans[0].body_parts[i].y)

                        # print("RShoulder:",humans[0].body_parts[i].x ," ",humans[0].body_parts[i].y)
                        # file.write("RShoulder:")
                        # file.write(str(humans[0].body_parts[i].x))
                        # file.write(" ")
                        # file.write("RX")
                        # file.write(str(humans[0].body_parts[i].y))
                        # file.write(" ")
                        # file.write("RY")

                    if i in [5]:
                        LShoulder.append(humans[0].body_parts[i].x)
                        LShoulder.append(humans[0].body_parts[i].y)

                        # print("LShoulder:",humans[0].body_parts[i].x ," ",humans[0].body_parts[i].y)
                        # file.write("LShoulder:")
                        # file.write(str(humans[0].body_parts[i].x))
                        # file.write(" ")
                        # file.write("LX")
                        # file.write(str(humans[0].body_parts[i].y))
                        # file.write(" ")
                        # file.write("LY")

                    # file.write(str(humans[0].body_parts[i].x))
                    # file.write(" ")
                    # file.write(str(humans[0].body_parts[i].y))
                    # file.write(" ")
                    cnt = cnt + 2

                    # storage.append(humans[0].body_parts[i].x)
                    # storage.append(humans[0].body_parts[i].y)
                    # print(storage)
                else:
                    # print(i,": ",0 ," ",0 )
                    # storage.append(0)
                    # storage.append(0)
                    # file.write(str(0))
                    # file.write (" ")
                    # file.write(str(0))
                    # file.write(" ")
                    cnt = cnt + 2

                if cnt % 36 is 0:
                    # print("LShoulder:",LShoulder,"RShoulder:",RShoulder,"LAnkle:",LAnkle,"RAnkle:",RAnkle)
                    if len(LShoulder)!=0 and len(RShoulder)!=0 and len(LAnkle)!=0 and len(RAnkle)!=0:
                        LenShoulder = LShoulder[0] - RShoulder[0]  # + LShoulder[1] - RShoulder[1]
                        # DShoulder = np.sqrt(np.square(LShoulder[0]) + np.square(RShoulder[0])) + np.sqrt(np.square(LShoulder[1]) + np.square(RShoulder[1]))

                        LenAnkle = LAnkle[0] - RAnkle[0]  # + LAnkle[1] - RAnkle[1]
                        # DAnkle = np.sqrt(np.square(LAnkle[0]) + np.square(RAnkle[0])) + np.sqrt(np.square(LAnkle[1]) + np.square(RAnkle[1]))

                        print("LenShoulder:", LenShoulder, " ", "LenAnkle:", LenAnkle)
                        # print("DShoulder:",DShoulder," ","DAnkle:",DAnkle)
                        file.write("LenShoulder:" + str(LenShoulder) + " " + "LenAnkle:" + str(LenAnkle))
                        file.write("\n")

        




                #np.savetxt("test2.txt", storage, fmt="%2.3f", delimiter=",")
        #while count < len(humans[0].body_parts):
        #        if(humans[0].body_parts[count]) is None:
        #            print("0")
        #        else:
        #            print(humans[0].body_parts[count].part_idx)
        #        count += 1
        #logger.debug('output_pts+')

    #np.savetxt("output", storage, newline=" ")
    file.close()
    cv2.destroyAllWindows()
