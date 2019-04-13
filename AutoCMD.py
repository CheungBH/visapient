import os

def GetVNum(strls):
    num=[]
    for i in strls:
        num.append(int(i[-5]))
    return max(num)+1

def GetPNum(strls):
    num=[]
    for i in strls:
        num.append(int(i[-1]))
    return max(num)

def GetInfoFromls(avils):
    actions=[]
    peoplestr=[]
    videostr=[]
    for i in avils:
        i=i.replace('\\', '/')
        actions.append(i.split('/')[-3])
        peoplestr.append(i.split('/')[-2])
        videostr.append(i.split('/')[-1])
        people=GetPNum(peoplestr)
        video=GetVNum(videostr)
    return list(set(actions)),people,video

def GetInfoFromPath(file_dir):
    AVIS=[]
    for dirpath,dirnames,filenames in os.walk(file_dir, topdown=False):
        for filename in filenames: #遍历所有有文件的文件夹
            #输出文件所在文件夹路径
            if filename[-4:] == '.avi':
                AVIS.append(dirpath+'/'+filename)
    actions,people,video = GetInfoFromls(AVIS)
    return actions,people,video

def GetInput():
    rawpath = input('please input the path (to the mainfolder):')
    realpath = rawpath.replace('\\', '/')
    Mfolder = realpath.split('/')[-1]
    activity,peo,vnum=GetInfoFromPath(realpath)
    frames=input('input your frame list (seperated by ","):')
    pts = input('input the process points:')
    md = eval(input('input the model sequence:  1-cmu_640x360,2-cmu_640x480, 3-mobilenet_thin_432x368 '))
    if md == 1:
        modelname = 'cmu_640x360'
    elif md == 2:
        modelname = 'cmu_640x480'
    else:
        modelname = 'mobilenet_thin_432x368'
    framelist = frames.split(",")
    framesint=[ int(x) for x in framelist ]
    return activity,peo,framesint,vnum,realpath,Mfolder,pts,modelname

def CreateNewDir(act,peo,framesls,path,pt,modelname):
    for m in framesls:
        for i in range(len(act)):
            for j in range(peo):
                NewPath = path + '/'+ str(pt) +'points/'+ modelname + '/data_' + str(m) + '/' + act[i] + '/p' + str(j + 1)
                os.makedirs(NewPath)
#创建新的data文件夹，用来装后续生成的txt文件

def ToDoc(act,peo,framesls,vnum,MFolder,pt,modelname):
    doc = open('out.txt','w')
    for k in framesls:
        for i in range((len(act))):
            for m in range(peo):
                for j in range(vnum):
                    h = str(j)
                    print('python src/{5}points/run_video_output{4}.py --video Video/TrainVideo/{3}/{0}/p{1}/{2}.avi --output Video/TrainVideo/{3}/{5}points/{6}/data_{4}/{0}/p{1}/{2}.txt  --model {6}'.format(act[i], m+1, h.zfill(2), MFolder,k,pt,modelname), file=doc)
    doc.close()
#循环创建全部指令，输出至文件
#循环创建全部指令，输出至文件

def ReadDoc():
    result=[]
    with open('out.txt','r') as f:
        for line in f.readlines():
            lines=line.strip('\n')
            result.append(lines)
    return result
#从文件中读入全部指令放入列表

def RunCMD(comm):
    for i in range(len(comm)):
        os.system(comm[i])
#运行生成的所有cmd指令，将文件输出至新生成的data文件夹中

def Gettxts(framelist,activity,peo,vnum,mpath):
    txtlist=[]
    for k in framelist:
        for i in activity:
            for j in range(peo):
                for m in range(vnum):
                    s=str(m)
                    filename=mpath + '/data_' + str(k) + '/' + i + '/p' + str(j + 1)+'/'+s.zfill(2)+'.txt'
                    txtlist.append(filename)                       #生成所有txt文件路径，并且存进一个列表中；返回列表的同时返回主路径名
    return txtlist

def deleteline(txtlist):
    for i in txtlist:       #对列表中的txt文件逐一进行读取
        readfile = open(i)
        lines = readfile.readlines()      #读入文件的某一行，并存入列表中
        readfile.close()
        w = open(i, 'w')
        w.writelines([item for item in lines[:-1]])       #将删除了最后一行的txt文件写回原文件
        w.close()

action,people,frame,videonum,path,MainFolder,points,model=GetInput()
CreateNewDir(action,people,frame,path,points,model)
ToDoc(action,people,frame,videonum,MainFolder,points,model)
cmd=ReadDoc()
RunCMD(cmd)
# txtname=Gettxts(frame,action,people,videonum,path)
# deleteline(txtname)
