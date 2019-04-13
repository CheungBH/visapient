import os

def GettxtPath(filedir):
    txtls=[]
    for dirpath,dirnames,filenames in os.walk(filedir):
        path=dirpath.replace('\\','/')
        for filename in filenames:
            if filename[-4:] == '.txt':
                txtls.append(path+'/'+filename)
    return txtls

def division(ls):
    dpath=[]
    lpath=[]
    for i in ls:
        if i[-9:-4] == 'label':
            lpath.append(i)
        else:
            dpath.append(i)
    return lpath,dpath

def FinalMerge(txtlist,sign):
    for filename in txtlist:     #逐一读取需要合并的文件
        finalpath ='/'.join(filename.split('/')[:-1]) + '/all/' + sign + '.txt'     #不同动作的文件合并进不同的最终文件中
        #print(finalpath)
        finalfile = open(finalpath, 'a')      #将最终文件打开为追加写模式
        for line in open(filename):
            finalfile.writelines(line)       #将源文件的每一行写入最终文件中
        finalfile.close()

MainPath=os.getcwd()
txtlist=GettxtPath(MainPath)
Labelpath,Datapath=division(txtlist)
os.makedirs('all')
FinalMerge(Labelpath,'label')
FinalMerge(Datapath,'data')
