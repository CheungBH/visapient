import numpy as np
import math

def GetSResult(RShoulder,RElbow,RWrist,LShoulder,LElbow,LWrist,RAnkle,LAnkle):
    result=[]
    ResultLeg=GetLegInfo(LShoulder,RShoulder,LAnkle,RAnkle)
    ResultLArm=GetArmInfo(LElbow,LWrist,LShoulder)
    ResultRArm=GetArmInfo(RElbow,RWrist,RShoulder)
    ResultHand=GetHandInfo(LWrist,RWrist,RElbow)
    result.append(ResultLeg)
    result.append(ResultLArm)
    result.append(ResultRArm)
    result.append(ResultHand)
    return result

def GetLegInfo(LShoulder,RShoulder,LAnkle,RAnkle):
    if len(LShoulder) == 2 and len(RShoulder) == 2 and len(LAnkle) == 2 and len(RAnkle) == 2:
        LenShoulder = LShoulder[0] - RShoulder[0]
        LenAnkle = LAnkle[0] - RAnkle[0]
        LegJudge = CheckLegDis(LAnkle, RAnkle, LShoulder, RShoulder)
        if LegJudge == 1:
            return "The leg is too Narrow"
        elif LegJudge == 2:
            return "The leg is too Wide"
        else:
            return "Suitable distance"

def GetArmInfo(Elbow,Wrist,Shoulder):
    if len(Elbow) == 2 and len(Shoulder) == 2 and len(Wrist) == 2:
        Angle = GetAngle(Elbow, Shoulder, Wrist)
        if CheckStraightArm(Elbow, Wrist, Shoulder) == False:
            return "Left Arm is too curve"
        else:
            return "Your left arm is right"


def GetHandInfo(LWrist,RWrist,RElbow):
    LenHand = CalDis(LWrist, RWrist)
    LenArm = CalDis(RWrist, RElbow)
    if CheckHand(LWrist, RWrist, RElbow) == True:
        return "Your hands' in a right position"
    else:
        return "The hands are divided"


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

def CheckStraightArm(coor1,coor2,coor3):
    angle=GetAngle(coor1,coor2,coor3)
    if angle<160:
        return False
    else:
        return True

def CheckLegDis(LeftAn,RightAn,LeftSh,RightSh):
    LenShoulder = LeftSh[0] - RightSh[0]
    LenAnkle = LeftAn[0] - RightAn[0]
    if LenAnkle<0.75*LenShoulder:
        return 1  #太窄
    elif LenAnkle>1.25*LenShoulder:
        return 2  #太宽
    else:
        return 0  #距离合适

def CheckHand(LHand,RHand,RElbow):
    HandDis=CalDis(LHand,RHand)
    ArmDis=CalDis(RHand,RElbow)
    if HandDis<0.6*ArmDis:
        return True
    else:
        return False