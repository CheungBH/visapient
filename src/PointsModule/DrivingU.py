import numpy as np
import math

def GetUResult(RShoulder,LShoulder,LElbow,LWrist,RAnkle,LAnkle):
    Result=[]
    ResultTurn=GetTResult(LShoulder,RShoulder,LAnkle,RAnkle)
    ResultArm=GetAResult(LElbow,LShoulder,LWrist)
    ResultHeight=GetHResult(LWrist,LShoulder)
    Result.append(ResultTurn)
    Result.append(ResultArm)
    Result.append(ResultHeight)
    return Result

def GetTResult(LShoulder,RShoulder,LAnkle,RAnkle):
    if len(LShoulder) == 2 and len(RShoulder) == 2 and len(LAnkle) == 2 and len(RAnkle) == 2:
        LenShoulder = LShoulder[0] - RShoulder[0]
        LenAnkle = LAnkle[0] - RAnkle[0]
        if CheckTurn(LAnkle, RAnkle, LShoulder, RShoulder) == True:
            return "Turning is right"
        else:
            return "You should turn more"

def GetAResult(LElbow,LShoulder,LWrist):
    if len(LElbow) == 2 and len(LShoulder) == 2 and len(LWrist) == 2:
        Angle = GetAngle(LElbow, LShoulder, LWrist)
        if CheckStraightArm(LElbow, LWrist, LShoulder) == False:
            return "Left Arm is too curve"
        else:
            return "Your left arm is right"

def GetHResult(LWrist,LShoulder):
    if LWrist[0] < LShoulder[0]:
        return "Your arm raised high enough"
    else:
        return "Your arm should be raised more"

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
    if angle<150:
        return False
    else:
        return True

def CheckTurn(LeftAn,RightAn,LeftSh,RightSh):
    LenShoulder = LeftSh[0] - RightSh[0]
    LenAnkle = LeftAn[0] - RightAn[0]
    if LenShoulder<0.5*LenAnkle:
        return True
    else:
        return False