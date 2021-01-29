
import tensorflow as tf


# step1 加载训练集和测试集合

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

 

 

# step2 创建模型

def create_model():

  return tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='softmax')

  ])

model = create_model()

 

# step3 编译模型 主要是确定优化方法，损失函数等

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

 

# step4 模型训练  训练一个epochs

model.fit(x=x_train,

          y=y_train,

          epochs=200,

          )

 

# step5 模型测试

loss, acc = model.evaluate(x_test, y_test)

print("train model, accuracy:{:5.2f}%".format(100 * acc))

 

# step6 保存模型的权重和偏置

model.save('my2_model.h5')  # creates a HDF5 file 'my_model.h5'

 

# step7 删除模型

del model  # deletes the existing model

 

 

# step8 恢复模型

# returns a compiled model

# identical to the previous one

restored_model = tf.keras.models.load_model('my2_model.h5')

 

# step9 测试模型

loss, acc = restored_model.evaluate(x_test, y_test)

print("Restored model, accuracy:{:5.2f}%".format(100 * acc))
