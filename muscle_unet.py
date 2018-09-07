import os

from keras.models import *  #从keras中导入所有模型
from keras.layers import *  #从keras中导入所有层
from keras.optimizers import *  #从keras中导入所有优化器
from keras import backend as keras  #导入keras后端
from keras import backend as k
import tensorflow as tf             #导入tensorflow

from keras.backend.tensorflow_backend import set_session  #导入设置会话(主要用于配置GPU)

#set gpu mermory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

PathH5='/media/root/4b360635-5bd4-4994-a7e9-27a696052ba0/1_muscle/muscle_h5/'  #训练数据的H5文件存放路径
model_save_Path='/media/root/4b360635-5bd4-4994-a7e9-27a696052ba0/1_muscle/muscle_model/'#训练模型保存路径
listFileH5 = [] #H5文件列表
model_name = "tf_model_" #保存模型添加的字符串
model_suffix = ".h5"

for root, sub_dirs, filelist in os.walk(PathH5):
    for filename in filelist:
        listFileH5.append(filename)
        #获取要训练的H5文件个数
        trainset_num = len(listFileH5)

#定义精度计算函数  y_true真实标签  y_pred预测值
def precision(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    true_positives = keras.sum(keras.round(keras.clip(y_true_f*y_pred_f,0,1)))
    predicted_positives = keras.sum(keras.round(keras.clip(y_pred_f,0,1)))
    precision =true_positives/(predicted_positives+keras.epsilon())
    return precision


#定义回报率计算函数
def recall(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    true_positives = keras.sum(keras.round(keras.clip(y_true_f*y_pred_f,0,1)))
    possible_positives = keras.sum(keras.round(keras.clip(y_true_f,0,1)))
    recall = true_positives/(possible_positives+keras.epsilon())
    return recall

def fmeasure(y_true,y_pred):
    p = precision(y_true,y_pred)
    r = recall(y_true,y_pred)
    fmeasure = (2*p*r)/(p+r)
    return fmeasure

def EuclideanLoss(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    loss = keras.sum(keras.square(y_true_f-y_pred_f))
    return loss


def EuclideanLossWithWeight(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    y = keras.abs(y_true_f-y_pred_f)
    all_one = keras.ones_like(y_true_f)
    y1_1 = keras.clip(y-0.15*all_one,-1,0)
    y1_sign = k.clip(-1*k.sign(y1_1),0,1)
    y1 = y1_sign*y

    y2_1 = k.clip(y-0.15*all_one,0,5)
    y2_2 = k.clip(y2_1-0.5*all_one,-1,0)
    y2_sign = k.clip(-1*k.sign(y2_2),0,1)
    y2 = y2_sign*y

    y3_1 = k.clip(y-0.5*all_one,0,5)
    y3_2 = k.clip(y3_1-0.8*all_one,-1,0)
    y3_sign = k.clip(-1*k.sign(y3_2),0,1)
    y3 = y3_sign*y

    y4_1 = k.sign(y-0.8*all_one)
    y4_sign = k.clip(y4_1,0,1)
    y4 = y4_sign*y

    y_final = 0.8 *y1 + 1.4*y2 + 1.6*y3 +1.8*y4

    loss = keras.sum(keras.square(y_final))

    return loss



#定义模型
def Net():
    #480x640
    input_1 = Input(shape=(480, 640, 3),name='input1')

    #480x640
    conv1 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_1)
    conv2 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    #240x320
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #240x320
    conv3 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv4 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    #120x160
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    #120x160
    conv5 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv6 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    #60x80
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    # 60x80
    #conv7 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    #conv8 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    #drop4 = Dropout(0.5)(conv8)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    #
    conv9 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv10 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    drop5 = Dropout(0.5)(conv10)
    #
    #up6 = Convolution2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    #merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    #conv11 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    #conv12 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

    #120x160
    up7 = Convolution2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge7 = merge([conv6, up7])
    conv13 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv14 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
    #
    up8 = Convolution2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv14))
    merge8 = merge([conv4, up8])
    conv15 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv16 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv15)
    #
    up9 = Convolution2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv16))
    merge9 = merge([conv2, up9])
    conv17 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv18 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv17)
    conv19 = Convolution2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv18)

    conv20 = Convolution2D(1, 1, activation='sigmoid')(conv19)
    #480*640
    model = Model(input=input_1, output=conv20)
    #编译 设置优化器损失函数性能评估函数 lr是学习率
    model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=['acc', precision, recall, fmeasure])

    return model

#如果这个py文件作为一个单独的程序运行则程序从这开始,如果这个文件以模型的形式被导入的话下面的程序不运行
if __name__ == "__main__":

    model = Net()
    #训练模型可以不用载入权重,也可以载入之前训练好的权重
    model.load_weights('/media/root/4b360635-5bd4-4994-a7e9-27a696052ba0/1_muscle/muscle_model/tf_model_100000.h5', by_name='TRUE')
    #打印模型概要
    print(model.summary())

    data_one = np.zeros([1, 480, 640, 3], dtype=float)
    label_one = np.zeros([1, 480, 640, 1], dtype=float)

    for num in range(0, 100000):
                                         #num % trainset_num是让测试集的数据被循环取用,在测试集中可以直接写num,因为每个数据只会被测试一次
        f = h5py.File(PathH5 + listFileH5[num % trainset_num])

        data = f['data'][:]   #获取训练图像
        label = f['label'][:] #获取标签图像

        data_one[0,:,:,:] = data
        label_one[0,:,:,0] = label
        f.close()

        #获取训练误差的标量值或标量值的列表,输入分别是训练数据和标签数据的np数组或np数组的列表
        result = model.train_on_batch(data_one,label_one)
        print(" iteration ", num + 1, " loss: ", result)       #打印结果 结果是在编译器中定义的

        if (num + 1) % 1000 == 0:  #训练1000的倍数次时保存模型
          str_num = str(num + 1)
          model.save(model_save_Path + model_name + str_num + model_suffix)