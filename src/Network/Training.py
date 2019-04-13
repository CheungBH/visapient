import keras

import IPython
import pandas as pd
import keras
import itertools
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Activation
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from kerasify import export_model
from keras.models import model_from_json
import os


a=


rawpath=os.getcwd()
path=rawpath.replace('\\','/')

if os.path.isdir(path+'/'+str(a)+'/model')==True:
    pass
else:
    os.makedirs(path+'/'+str(a)+'/model')
    
if os.path.isdir(path+'/'+str(a)+'/graph')==True:
    pass
else:
    os.makedirs(path+'/'+str(a)+'/graph')


timesteps = 15
## POSE - val_acc: 98%
epochs = 4000
batch_size = 32
_dropout = 0.05
_activation='relu'
_optimizer='Adam'
class_names = ["FRight","FWrong","SRight","SWrongHand","UNoTurrn","URight","UWrongArm","UWrongFoot","WNoTurn","WPoor","WRight","WWrongFoot"] 
X_vector_dim = 36 # number of features or columns (pose)
samples_path = path+'/'+str(a)+"/data.txt" #311 files with 10 frames' human-pose estimation keypoints(10*18)
labels_path = path+'/'+str(a)+"/label.txt" #311 files' labels, 3 classes in total
model_path = path+'/'+str(a)+'/model/pose.model'
json_model_path = path+'/'+str(a)+'/model/pose_model.json'
model_weights_path = path+'/'+str(a)+"/model/pose_model.h5"

X = np.loadtxt(samples_path, dtype="float")
y = np.loadtxt(labels_path)

def samples_to_3D_array(_vector_dim, _vectors_per_sample, _X):
    X_len = len(_X)
    result_array = []
    for sample in range (0,X_len): #should be the 311 samples?
        sample_array = []
        for vector_idx in range (0, _vectors_per_sample):
            start = vector_idx * _vector_dim
            end = start + _vector_dim
            sample_array.append(_X[sample][start:end])
        result_array.append(sample_array)
    return np.asarray(result_array)

X_vectors_per_sample = timesteps # number of vectors per sample , 5 samples
X_3D = samples_to_3D_array(X_vector_dim, X_vectors_per_sample, X)

def convert_y_to_one_hot(_y): #one hot encoding simply means : red --> 0 , green --> 1 , blue --> 2
    _y = np.asarray(_y,dtype=int)
    b = np.zeros((_y.size, _y.max()+1))
    b[np.arange(_y.size),_y] = 1
    return b

y_one_hot = convert_y_to_one_hot(y)
y_vector_dim = y_one_hot.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_3D, y_one_hot, test_size=0.2, random_state=42)
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential()
model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
model.add(Dropout(_dropout))
model.add(TimeDistributed(Dense(X_vector_dim*2, activation=_activation))) #(5, 80)
model.add(Dropout(_dropout))
model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation))) #(5, 40)
model.add(Dropout(_dropout))
model.add(TimeDistributed(Dense(int(X_vector_dim/2), activation=_activation))) #(5, 20)
model.add(Dropout(_dropout))
model.add(TimeDistributed(Dense(int(X_vector_dim/4), activation=_activation))) #(5, 10)
model.add(Dropout(_dropout))
model.add(LSTM(int(X_vector_dim/4), dropout=_dropout, recurrent_dropout=_dropout))
model.add(Dense(y_vector_dim,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])

class TrainingVisualizer(keras.callbacks.History):
    def on_epoch_end(self, epoch, logs={}):
        super(TrainingVisualizer, self).on_epoch_end(epoch, logs)
        IPython.display.clear_output(wait=True)
        axes = pd.DataFrame(self.history).plot()
        axes.axvline(x=max((val_acc, i) for i, val_acc in enumerate(self.history['val_acc']))[1])
        #plt.show()
        

print('Training...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))
          #callbacks=[TrainingVisualizer()])

score, accuracy = model.evaluate(X_test, y_test,
                                 batch_size=batch_size)
#输出正确率
doc = open(path+'/'+str(a)+'/out.txt','w')
print('Test score: {:.3}'.format(score),file = doc)
print('Test accuracy: {:.3}'.format(accuracy),file = doc)



y_pred = model.predict(X_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path+'/'+str(a)+'/Confusion_Matrix')
    plt.close()

# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

#plt.show()

export_model(model,model_path)
print("Model saved to disk")

model_json = model.to_json()
with open(json_model_path, "w") as json_file:
    json_file.write(json_model_path)
# serialize weights to HDF5
model.save_weights(model_weights_path)
print("Saved model to disk")

os.system('cd ..')



