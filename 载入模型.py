# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:31:52 2019

@author: ThinkPad
"""
import tensorflow as tf


import matplotlib.pyplot as plt
import numpy as np

 

 

# step1 加载训练集和测试集合

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_test = x_test / 255

 
restored_model = tf.keras.models.load_model('my2_model.h5')

 

# step9 测试模型

loss, acc = restored_model.evaluate(x_test, y_test)

print("Restored model, accuracy:{:5.2f}%".format(100 * acc))

i = 23
plt.imshow(x_test[i],cmap=plt.cm.binary)

plt.show()

 

predictions = restored_model.predict(x_test)

print(np.argmax(predictions[i]))