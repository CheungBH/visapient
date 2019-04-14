import os

def GetInfoFromls(avils):
    actions=[]
    for i in avils:
        i=i.replace('\\', '/')
        actions.append(i.split('/')[-2])
    return list(set(actions))

def GetInfoFromPath(file_dir):
    AVIS=[]
    for dirpath,dirnames,filenames in os.walk(file_dir, topdown=False):
        for filename in filenames: #遍历所有有文件的文件夹
            #输出文件所在文件夹路径
            if filename[-4:] == '.jpg':
                AVIS.append(dirpath+'/'+filename)
    actions=GetInfoFromls(AVIS)
    return actions

def GetInput():
    rawpath = input('please input the path (to the mainfolder):')
    realpath = rawpath.replace('\\', '/')
    Mfolder = realpath.split('/')[-1]
    activity = GetInfoFromPath(realpath)
    pts = input('input the process points:')
    vnum = input("Input the number of pics: ")
    md = eval(input('input the model sequence:  1-cmu_640x360,2-cmu_640x480, 3-mobilenet_thin_432x368 '))
    if md == 1:
        modelname = 'cmu_640x360'
    elif md == 2:
        modelname = 'cmu_640x480'
    else:
        modelname = 'mobilenet_thin_432x368'
    return activity,vnum,realpath,Mfolder,pts,modelname

def CreateNewDir(act,path,pt,modelname):
    for i in range(len(act)):
        NewPath = path + '/'+ str(pt) +'points/'+ modelname + '/' + act[i]
        try:
            os.makedirs(NewPath)
        except FileExistsError:
            pass
#创建新的data文件夹，用来装后续生成的txt文件

def ToDoc(act,vnum,MFolder,pt,modelname):
    doc = open('out_image.txt','w')
    k=1
    for i in range((len(act))):
        for m in range(1):
            for j in range(int(vnum)):
                h = str(j)
                print('python src/{5}points/run_figure.py --image {3}/{0}/frame{2}.jpg --output {3}/{5}points/{6}/{0}/frame{2}.txt  --model {6}'.format(act[i], m+1, h, MFolder,k,pt,modelname), file=doc)
    doc.close()

def ReadDoc():
    result=[]
    with open('out_image.txt','r') as f:
        for line in f.readlines():
            lines=line.strip('\n')
            result.append(lines)
    return result

def RunCMD(comm):
    for i in range(len(comm)):
        try:
            os.system(comm[i])
        except FileNotFoundError:
            pass


action,videonum,path,MainFolder,points,model=GetInput()
CreateNewDir(action,path,points,model)
ToDoc(action,videonum,MainFolder,points,model)
cmd=ReadDoc()
RunCMD(cmd)
