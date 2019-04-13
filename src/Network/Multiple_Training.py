import os

a=eval(input("Begin from:"))
b=eval(input("End with:"))
cmdls=[]

for i in range(a,b+1):
    pycmd='python '+str(i)+'/training.py'
    cmdls.append(pycmd)

for i in cmdls:
    os.system(i)