import os
import pandas as pd

def FindWrong(csvfile,rawpath):
    path=rawpath.replace('\\','/')
    file=pd.read_csv(csvfile)
    WrongMessage=file.values.tolist()
    dellist=[]
    waitlist=[]
    for i in WrongMessage:
        Wrongtxt=path + '/' + i[1] + '/' + i[2] + '/' + str(i[3]).zfill(2) + '.txt'
        if i[4] == 'del':
            dellist.append(Wrongtxt)
        else:
            waitlist.append(Wrongtxt)
    return dellist, waitlist

def DeleteFile(FileList):
    for i in FileList:
        try:
            os.remove(i)
        except FileNotFoundError:
            continue

Mainpath=os.getcwd()

csvPath=input("Please input the path of the csv:")         #输入包含错误信息的csv文件路径

waitlist=FindWrong(csvPath,Mainpath)

dellist, waitlist = FindWrong(csvPath,Mainpath)