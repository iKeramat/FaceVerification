import tensorflow as tf
from tensorflow.keras import layers as ksl
import numpy as np
import sklearn as sk
import numpy as np
from matplotlib import pyplot as plt

class Model:
    def __init__(self):
        self.net =  self.build()
        pass

    def build(self):
        inp = ksl.Input([64,64,3])
        x = ksl.Conv2D(64,(3,3),padding='same',activation='relu')(inp)
        x = ksl.MaxPool2D(pool_size=[2,2])(x)
        x = ksl.Dropout(0.3)(x)

        x = ksl.Conv2D(64,(2,2),padding='same',activation='relu')(x)
        x = ksl.MaxPool2D(pool_size=[2,2])(x)
        x = ksl.Dropout(0.3)(x)
        poolOut = ksl.GlobalAveragePooling2D()(x)
        x = ksl.Dense(256)(poolOut)
        model = tf.keras.Model(inp,x)

        im1 = ksl.Input([64,64,3])
        im2 = ksl.Input([64,64,3])
        feature1 = model(im1)
        feature2 = model(im2)
        distance = ksl.Lambda(self.euclidean_distance)([feature1,feature2])
        net = tf.keras.Model([im1,im2],distance)

        return net

    def compileModel(self):
        self.net.compile(loss='binart_crossentropy',optimizer='adam',metrics=['accuracy'])


    def train(self,images,target):
        self.net.fit(images,self.target,epochs=10,batch_size=32)

    @staticmethod
    def plotHistory(Hist):
        # plot History
        plt.plot(Hist.history['accuracy'])
        plt.plot(Hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.show()
        plt.plot(Hist.history['loss'])
        plt.plot(Hist.history['val_loss'])
        plt.title('model loss')
        plt.show()

    @staticmethod
    def euclidean_distance(vects):
        yA,yB = vects
        return tf.math.reduce_euclidean_norm(yA-yB,axis = 1,keepdims = True)

    @staticmethod
    def controstivLoss(m = 1.0):
        def loss(yTrue,yPred):
            squarePred = tf.square(yPred)
            l = yTrue*squarePred+(1-yTrue)*tf.square(tf.maximum(0.0,m-squarePred))
            return tf.reduce_mean(l)
        return loss


