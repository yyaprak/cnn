'''
https://www.youtube.com/watch?v=uqomO_BZ44g
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
#from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import numpy as np

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.models import layers


img=image.load_img("training/bozuk_yol/3.jpeg")

plt.imshow(img)

print(cv2.imread("training/bozuk_yol/3.jpeg").shape)
print(img)

training =ImageDataGenerator(rescale=1/255)
validation =ImageDataGenerator(rescale=1/255)





training_dataset=training.flow_from_directory('training/',target_size=(200,200),
                                        batch_size=3,
                                        class_mode='binary'
                                        )
validation_dataset=training.flow_from_directory('training/',target_size=(200,200),
                                        batch_size=3,
                                        class_mode='binary'
                                        )
model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  #
                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  #
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  ##
                                  tf.keras.layers.Flatten(),
                                  ##
                                  tf.keras.layers.Dense(512,activation='relu'),
                                  ##
                                  tf.keras.layers.Dense(1,activation='sigmoid')
                                  ])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
history_model_fit=model.fit(training_dataset,
                    steps_per_epoch=3,
                    epochs=32,
                    validation_data=validation_dataset)

#array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

dir_path='testing'
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    #print(i)
    
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val==0:
        print("yol bozuk")
    else:
        print("yol duzgun")
    
    
print(training_dataset.class_indices)

#
#accuracy and losses graphs   


acc = history_model_fit.history['accuracy']
val_acc = history_model_fit.history['val_accuracy']
loss = history_model_fit.history['loss']
val_loss = history_model_fit.history['val_loss']
plt.figure(figsize=(8, 8))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.show()


plt.figure(figsize=(8, 8))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
