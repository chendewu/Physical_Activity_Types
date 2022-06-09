import joblib as joblib
import pandas as pd
import matplotlib.pyplot as plt
import keras
import math

from keras import layers
from keras.models import Model, Sequential
from keras.layers import Conv1D, Embedding, CuDNNLSTM, LeakyReLU
from keras.layers import Dropout
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import GlobalMaxPool1D
from keras.layers import Bidirectional
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import Add, Concatenate
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import keras.backend as K
import numpy as np
import tensorflow as tf

import os
from sklearn.model_selection import train_test_split
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.model_selection import KFold

load_data_1_5 = np.load("dataset_for_suppl_1_5.npy", allow_pickle=True)
load_data_all = load_data_1_5[0]
for i in range(1, load_data_1_5.shape[0]):
    load_data_all = np.vstack((load_data_all, load_data_1_5[i]))
load_data_6_10 = np.load("Dataset1_for_suppl_6_10.npy", allow_pickle=True)
for i in range(0, load_data_6_10.shape[0]):
    load_data_all = np.vstack((load_data_all, load_data_6_10[i]))

'''
minValue = np.min(load_data_all)
maxValue = np.max(load_data_all)
load_data_all = (load_data_all - minValue) / (maxValue - minValue)
'''

'''
avgValue = np.average(load_data_all)
stdValue = np.std(load_data_all)
load_data_all = (load_data_all - avgValue) / stdValue
'''

label_all = []
for i in range(0, load_data_1_5.shape[0]):
    for j in range(0, load_data_1_5[i].shape[0]):
        label_all.append(i)
for m in range(0, load_data_6_10.shape[0]):
    for n in range(0, load_data_6_10[m].shape[0]):
        label_all.append(m + load_data_1_5.shape[0])
label_all = np.asarray(label_all, np.int32)
label_all = keras.utils.to_categorical(label_all, 10)

# 打乱顺序
num_example_train = load_data_all.shape[0]
arr_train = np.arange(num_example_train)
if (os.path.exists("filename.txt") == False):
    np.random.shuffle(arr_train)
    np.savetxt("filename.txt", arr_train)
else:
    arr_train =  np.loadtxt("filename.txt", dtype=int)
load_data_all = load_data_all[arr_train]
label_all = label_all[arr_train]

batch_size_train = 64
#s_train = int(num_example_train * 0.98)
kf = KFold(n_splits=10,shuffle=False)  # 初始化KFold
train_index = []
test_index = []
i = 0
for train_index_temp, test_index_temp in kf.split(arr_train):  # 调用split方法切分数据
    print('train_index:%s , test_index: %s ' % (train_index_temp, test_index_temp))
    train_index = train_index_temp
    test_index = test_index_temp
    i = i + 1
    if i == 9:
        break
train_data, train_label = load_data_all[train_index], label_all[train_index]
test_data, test_label = load_data_all[test_index], label_all[test_index]

input_shape = (128, 3)
model = Sequential()
model.add(Conv1D(32, 3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(256, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

from keras.utils import plot_model
#plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
def show_train_acc(y1, label1, y2, label2):
    # 画图
    plt.title('train acc')  # 标题
    # 常见线的属性有：color,label,linewidth,linestyle,marker等
    plt.plot(y1, color='cyan', label=label1)
    plt.plot(y2, 'b', label=label2)  # 'b'指：color='blue'
    plt.legend()  # 显示上面的label
    plt.xlabel('Epoch')
    plt.show()

def show_train_loss(y3, label3, y4, label4):
    # 画图
    plt.title('train loss')  # 标题
    plt.plot(y3, 'r', label=label3)  # 'b'指：color='blue'
    plt.plot(y4, 'g', label=label4)  # 'b'指：color='blue'
    plt.legend()  # 显示上面的label
    plt.xlabel('Epoch')
    plt.show()

def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
   print(lrate)
   return lrate

#Custom callback
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
class MyCallBack(EarlyStopping):
        def __init__ ( self , training_data , validation_data, verbose ):
            self.x = training_data
            self.y_val = validation_data

        def on_train_begin ( self , logs={} ):
            return

        def on_train_end ( self , logs={} ):
            return

        def on_epoch_begin ( self , epoch , logs={} ):
            return

        def on_epoch_end ( self , epoch , logs={} ):
            y_pred = self.model.predict ( self.x )

            roc = roc_auc_score ( self.y_val , y_pred )

            roc_val = roc_auc_score ( self.y_val , y_pred )
            print ( '\rroc-auc: %s - roc-auc_val: %s' % (str ( round ( roc , 4 ) ) , str ( round ( roc_val , 4 ) )) ,
                    end=100 * ' ' + '\n' )

            test = [ 0 ] * y_pred.shape[0]
            i = 0
            for cl in y_pred:
                test[ i ] = str ( np.argmax ( cl ) )
                i += 1

            test_lab = [ 0 ] * y_pred.shape[0]
            i = 0
            for cl in self.y_val:
                test_lab[ i ] = str ( np.argmax ( cl ) )
                i += 1

            print(classification_report(test , test_lab ))

            acc = accuracy_score ( test , test_lab )
            print ( " Acc:" , acc )

            pre = precision_score ( test , test_lab , average='micro' , labels=[ '1' , '2' , '3' ] )
            print ( " Pre:" , pre )

            re = recall_score ( test , test_lab , average='micro' , labels=[ '1' , '2' , '3' ] )
            print ( " Rec:" , re )

            f1 = f1_score ( test , test_lab , average='micro' , labels=[ '1' , '2' , '3' ] )
            print ( " F1:" , f1 )

            return

        def on_batch_begin ( self , batch , logs={} ):
            return

        def on_batch_end ( self , batch , logs={} ):
            return

filepath1="best_model.{epoch:02d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath=filepath1, monitor='accuracy', save_best_only=False)

callbacks_list = [MyCallBack(test_data, test_label,verbose=1),
                            checkpoint,
                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                            keras.callbacks.LearningRateScheduler(step_decay),]

adam = keras.optimizers.Adam(lr=0.001)
model.compile ( loss='categorical_crossentropy' , optimizer=adam , metrics=['accuracy'] )
history = model.fit(train_data, train_label, batch_size_train, 300,
                     callbacks=callbacks_list,
                     validation_data = [test_data, test_label],
                      verbose=1)
np.savetxt("acc.txt", history.history['accuracy'], fmt='%0.8f', delimiter=',')
np.savetxt("loss.txt", history.history['loss'], fmt='%0.8f', delimiter=',')
np.savetxt("val_acc.txt", history.history['val_accuracy'], fmt='%0.8f', delimiter=',')
np.savetxt("val_loss.txt", history.history['val_loss'], fmt='%0.8f', delimiter=',')
show_train_acc(history.history['accuracy'], 'train_accuracy', history.history['val_accuracy'], 'val_accuracy')
show_train_loss(history.history['loss'], 'train_loss', history.history['val_loss'], 'val_loss')