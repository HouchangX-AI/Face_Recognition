# 导入包
import os
import keras
import numpy as np
from keras import optimizers
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,load_model,Model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout

from Config import config
from Get_Data import DataSet


#建立一个基于CNN的人脸识别模型
class MY_Model(object):
    FILE_PATH = config['model_file_path']   #模型进行存储和读取的地方
    IMAGE_SIZE = config['image_size']

    def __init__(self):
        self.model = None

    #读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self,dataset):
        self.dataset = dataset

    #建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_basic_cnn_model(self, file_path=FILE_PATH):
        if os.path.exists(file_path):
            self.model = load_model(file_path)
            self.model.summary()
        else:
            self.model = Sequential()
            self.model.add(Convolution2D(filters=32,kernel_size=(3, 3),padding='same',input_shape=self.dataset.X_train.shape[1:]))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2),padding='same'))
            
            self.model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

            self.model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
            
            self.model.add(Flatten())
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))

            self.model.add(Dense(self.dataset.num_classes))
            self.model.add(Activation('softmax'))
            self.model.summary()

    def build_MobileNet_model(self, file_path=FILE_PATH):
        if os.path.exists(file_path):
            self.model = load_model(file_path)
            self.model.summary()
        else:
            # self.model = keras.applications.mobilenet.MobileNet(include_top=True,weights=None,classes=self.dataset.num_classes)
            base_model = keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3),pooling='avg')
            image_input = base_model.input
            x = base_model.layers[-1].output
            x = Dense(self.dataset.num_classes)(x)
            out = Activation('softmax')(x)
            new_model = Model(image_input, out)
            self.model = new_model
            self.model.summary()

    #进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self, file_path=FILE_PATH):
        adam = optimizers.Adam(lr=0.0001)
        self.model.compile(
            optimizer=adam,  #有很多可选的optimizer，例如RMSprop,Adagrad，你也可以试试哪个好，我个人感觉差异不大
            # loss='categorical_crossentropy',  #你可以选用squared_hinge作为loss看看哪个好
            loss='squared_hinge',
            metrics=['accuracy'])

        #epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
        # model_names = file_path + 'model.{epoch:02d}-{val_loss:.4f}.h5'
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True)
        self.model.fit(self.dataset.X_train,
                        self.dataset.Y_train,
                        epochs=config['eopch'],
                        batch_size=config['batch_size'],
                        validation_split=config['validation_split'],
                        callbacks=[model_checkpoint])

    def evaluate_model(self, file_path=FILE_PATH):
        self.model = load_model(file_path)
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)



if __name__ == '__main__':
    print('get dataset')
    dataset = DataSet('Files/data_50man_300p_224')
    print('get model')
    model = MY_Model()
    print('read trainData')
    model.read_trainData(dataset)
    print('build model')
    #model.build_basic_cnn_model()
    model.build_MobileNet_model()
    print('train')
    model.train_model()
    model.evaluate_model()
    # model.save()




