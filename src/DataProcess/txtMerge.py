import os

def Findtxt(filedir):
    txtls=[]
    for dirpath,dirnames,filenames in os.walk(filedir):
        path=dirpath.replace('\\','/')
        for filename in filenames:
            if filename[-4:] == '.txt':
                txtls.append(path+'/'+filename)
    return txtls

def txtMerge(txtls):
    for filename in txtls:     #逐一读取需要合并的文件
        finalpath = '/'.join(filename.split('/')[:-3]) + '/Raw/' + filename.split('/')[-3] + '.txt'     #不同动作的文件合并进不同的最终文件中
        finalfile = open(finalpath, 'a')      #将最终文件打开为追加写模式
        for line in open(filename):
            finalfile.writelines(line)       #将源文件的每一行写入最终文件中
        finalfile.close()

MainPath=os.getcwd()
txtnames=Findtxt(MainPath)
os.makedirs('Raw')
txtMerge(txtnames)
