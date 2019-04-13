import numpy as np
import math

def GetFResult(RShoulder,RElbow,LShoulder,RAnkle,LAnkle):
    Result=[]
    ResultTurn=GetTResult(LShoulder,RShoulder,LAnkle,RAnkle)
    ResultHeight=GetHResult(RElbow,RShoulder)
    Result.append(ResultTurn)
    Result.append(ResultHeight)
    return Result

def GetTResult(LShoulder,RShoulder,LAnkle,RAnkle):
    LenShoulder = LShoulder[0] - RShoulder[0]
    LenAnkle = LAnkle[0] - RAnkle[0]
    if CheckTurn(LAnkle, RAnkle, LShoulder, RShoulder) == True:
        return "Turning is right"
    else:
        return "You should turn more"

def GetHResult(RElbow,RShoulder):
    if RElbow[1] < RShoulder[1] * 1.02:
        return "Excellent Finish"
    else:
        return "Your arm should raise more"

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


def CheckTurn(LeftAn,RightAn,LeftSh,RightSh):
    LenShoulder = LeftSh[0] - RightSh[0]
    LenAnkle = LeftAn[0] - RightAn[0]
    if LenShoulder<0.4*LenAnkle:
        return True
    else:
        return False