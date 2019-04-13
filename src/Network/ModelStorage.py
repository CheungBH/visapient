
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Activation
from keras.layers import LSTM


def GetModel(No,_dropout,X_vector_dim,_activation,input_shape,y_vector_dim):
    if No==0:
        model,des = Model0(_dropout,X_vector_dim,_activation,input_shape,y_vector_dim)
    elif No==1:
        model, des = Model1(_dropout, X_vector_dim, _activation, input_shape,y_vector_dim)
    elif No==2:
        model, des = Model2(_dropout, X_vector_dim, _activation, input_shape,y_vector_dim)
    elif No==3:
        model, des = Model3(_dropout, X_vector_dim, _activation, input_shape,y_vector_dim)
    else:
        print("Model's not enough")
    return model, des

def Model0(_dropout,X_vector_dim,_activation,input_shape,y_vector_dim):
    model = Sequential()
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation)))  # (5, 40)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(int(X_vector_dim / 4), activation=_activation)))  # (5, 10)
    model.add(Dropout(_dropout))
    model.add(LSTM(int(X_vector_dim / 4), dropout=_dropout, recurrent_dropout=_dropout))
    model.add(Dense(y_vector_dim, activation='softmax'))
    NetworkInfo = '*2_*1_/2_/4'
    return model, NetworkInfo

def Model1(_dropout,X_vector_dim,_activation,input_shape,y_vector_dim):
    model = Sequential()
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation)))  # (5, 40)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(int(X_vector_dim / 4), activation=_activation)))  # (5, 10)
    model.add(Dropout(_dropout))
    model.add(LSTM(int(X_vector_dim / 4), dropout=_dropout, recurrent_dropout=_dropout))
    model.add(Dense(y_vector_dim, activation='softmax'))
    NetworkInfo = '*2_*2_*1_/2_/4'
    return model, NetworkInfo

def Model2(_dropout,X_vector_dim,_activation,input_shape,y_vector_dim):
    model = Sequential()
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation)))  # (5, 40)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
    model.add(Dropout(_dropout))
    model.add(LSTM(int(X_vector_dim / 2), dropout=_dropout, recurrent_dropout=_dropout))
    model.add(Dense(y_vector_dim, activation='softmax'))
    NetworkInfo = '_*2_*1_/2'
    return model, NetworkInfo

def Model3(_dropout,X_vector_dim,_activation,input_shape,y_vector_dim):
    model = Sequential()
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim * 4, activation=_activation)))  # (5, 80)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation)))  # (5, 40)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
    model.add(Dropout(_dropout))
    model.add(TimeDistributed(Dense(int(X_vector_dim / 4), activation=_activation)))  # (5, 10)
    model.add(Dropout(_dropout))
    model.add(LSTM(int(X_vector_dim / 4), dropout=_dropout, recurrent_dropout=_dropout))
    model.add(Dense(y_vector_dim, activation='softmax'))
    NetworkInfo = '_*2_*1_/2'
    return model, NetworkInfo
