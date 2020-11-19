########### import 模型用到的 package ###########
import numpy as np
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras import regularizers
import tensorflow as tf
import os

################### GPU運算設定 ###################
# 指定第幾張 GPU 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 自動增長 GPU 記憶體用量
#gpu_options = tf.GPUOptions(allow_growth=True)
 
# 只使用 xx% 的 GPU 記憶體
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 Session
tf.keras.backend.set_session(sess)


################# load data ###################
protein = 'EGFR_family'
data_fp = pd.read_csv('data/EGFR_family_cm_f34_half_test_all_act.csv', header=None)

x_feature = data_fp.values[:, 2::]
y_label = data_fp.values[:, 0:2]
print("---------------------- Data information ----------------------")
print('EGFR_data:\n',data_fp.head(10))
print('x_feature:', x_feature.shape)
print('y_label:', y_label.shape)

############################## plot figure ###############################
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plot_roc_curve(fpr,tpr): 
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC curve')
    plt.show()

################################## result list ####################################

cv_acc = []
cv_sen = []
cv_spe = []
cv_auc = []
cv_mcc = []
cv_f1 = []

random_seed = 2
cv_n = 1

######################### 劃分 train跟test data #########################
train_x, test_x, train_smile_y, test_smile_y = train_test_split(x_feature, y_label, train_size=0.8, test_size=0.2, random_state=random_seed)
train_y = train_smile_y[:,1]
test_y = test_smile_y[:,1]

train_x = tf.keras.backend.cast_to_floatx(train_x)
test_x = tf.keras.backend.cast_to_floatx(test_x)
train_y = tf.keras.backend.cast_to_floatx(train_y)
test_y = tf.keras.backend.cast_to_floatx(test_y)


################ print出data基本資料 ################

print('all:', x_feature.shape)
print('train_x:', train_x.shape)
print('test_x:', test_x.shape)
label_set = train_y.tolist()
#label_set = label_set.reset_index(drop=True)
label_count1 = label_set.count(1)
label_count0 = label_set.count(0)
print("##train:")
print('p:',label_count1)
print('n:',label_count0)

label_set = test_y.tolist()
#label_set = label_set.reset_index(drop=True)
label_count1 = label_set.count(1)
label_count0 = label_set.count(0)
print("##test:")
print('p:',label_count1)
print('n:',label_count0)

label_set = y_label[:,1].tolist()
#label_set = label_set.reset_index(drop=True)
label_count1 = label_set.count(1)
label_count0 = label_set.count(0)
print("##all:")
print('p:',label_count1)
print('n:',label_count0)
print('pos_rate:',round(label_count1/label_count0,3))

############## creat model & training #############

print("\n--------------------- Taining history ---------------------")
model = Sequential()  # 宣告keras model

model.add(Dense(1024, input_dim=238, kernel_initializer='uniform')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(768, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dropout(0.2))

model.add(Dense(512, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dropout(0.2))

model.add(Dense(256, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dropout(0.2))
model.add(Dense(units=1, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01)
               , activation='sigmoid'))

Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']) 

#################### set learning rate #####################
lrate = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=5, 
    verbose=0, 
    mode='auto', 
    epsilon=0.0001, 
    cooldown=0, 
    min_lr=0.00001 )

#################### training model #####################
train_history = model.fit(x=train_x,
              y=train_y,
              validation_data = (test_x,test_y),
              epochs=100,
              batch_size=64,
              verbose = 2,
              callbacks = [lrate])

print("check2:",cv_n)
show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')

############## Save model ################
save_model = "save_model/" + protein + "_DNN_model_cm_ac_fp.hdf5"
model.save( save_model )
print("------------------------- Save model -------------------------")
print("file:" + save_model,"\n")


################ Evalutation model ################
print("---------------------- Evalutation model ---------------------")
test_loss, test_acc = model.evaluate(test_x, test_y)
print("check4:",cv_n)

print ("** test loss: {}".format(test_loss))
print ("** test accuracy: {}".format(test_acc))

pred_prob = model.predict(test_x)     ##模型預測
pred_class = model.predict_classes(test_x)     
pred_class_train = model.predict_classes(train_x)     
y_test = test_y.astype(float)

fpr, tpr, thresholds = roc_curve(y_test, pred_prob, pos_label=1) ##計算ROC
plot_roc_curve(fpr,tpr)

print ("\nAcc : {}".format(round(test_acc,3)))

cm1 = confusion_matrix(y_test,pred_class)
sensitivity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])  ##計算sen
print('Sen : ', round(sensitivity1, 3))

specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])  ##計算spe
print('Spe : ', round(specificity1,3))

auc_score = roc_auc_score(y_test, pred_prob)  ##計算auc
print('AUC :', round(auc_score,3))

mcc_score = metrics.matthews_corrcoef(y_test, pred_class, sample_weight=None)  ##計算mcc
print('MCC :', round(mcc_score,3))

f1_score = metrics.f1_score(y_test, pred_class)  ##計算f1_score
print('f1 :', round(f1_score,3))

print('Confusion_Matrix : \n', cm1,'\n')

cv_acc.append(round(test_acc,3))
cv_sen.append(round(sensitivity1, 3))
cv_spe.append(round(specificity1,3))
cv_auc.append(round(auc_score,3))
cv_mcc.append(round(mcc_score,3))
cv_f1.append(round(f1_score,3))


################ loading 訓練好的 model ################
print("------------------------ Loading model ------------------------")
from keras.models import load_model
model_name = protein + "_DNN_model_cm_ac_fp.hdf5"
model_load = "save_model/" + model_name
print("**model_name:",model_load)
model = load_model(model_load)

test_loss, test_acc = model.evaluate(test_x, test_y)

print ("**load model acc : {}".format(test_acc))


    