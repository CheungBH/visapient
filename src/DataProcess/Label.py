import os

def GettxtPath(filedir):
    txtls=[]
    for dirpath,dirnames,filenames in os.walk(filedir):
        path=dirpath.replace('\\','/')
        for filename in filenames:
            if filename[-4:] == '.txt':
                txtls.append(path+'/'+filename)
    return txtls

def GetLabelPath(fls):
    lls=[]
    for i in fls:
        lpath='/'.join(i.split('/')[:-1]) + '/' + i.split('/')[-1][:-4] + '_label.txt'
        lls.append(lpath)
    return lls

def WriteLabel(fls,lls):
    for i in range(len(fls)):
        f=open(fls[i])
        n=len(f.readlines())
        f.close()
        flabel=open(lls[i],'w')
        for m in range(n):
            flabel.writelines(str(int((i+1))))
            flabel.writelines('\n')
        flabel.close()

MainPath = os.getcwd()

FinalList=GettxtPath(MainPath)      #得到所有整合后的文件路径列表

LabelList=GetLabelPath(FinalList)       #得到整合后的数据集对应的标签路径

WriteLabel(FinalList,LabelList)
