#!/usr/bin/env python
# coding: utf-8

# ### Import all dependencies

# In[28]:


from __future__ import print_function, division
from builtins import range, input

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import pandas as pd


# ### Define Few Parameters

# In[29]:


#define size to which images are to be resized
IMAGE_SIZE = [224, 224] # feel free to change depending on dataset

# training config:
epochs = 100
batch_size = 32

#define paths
TB_path = '/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/imgs/tb'
nonTB_path = '/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/imgs/test'

# Use glob to grab images from path .jpg or jpeg
TB_files = glob(TB_path + '/*')
nonTB_files = glob(nonTB_path + '/*')


# ### Fetch Images and Class Labels from Files (This might take a while)

# In[30]:


# Preparing Labels
TB_labels = []
nonTB_labels = []

TB_images=[]
nonTB_images=[]

import cv2 

for i in range(len(TB_files)):
  image = cv2.imread(TB_files[i])
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  TB_images.append(image)
  TB_labels.append('tb')
for i in range(len(nonTB_files)):
  image = cv2.imread(nonTB_files[i])
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  nonTB_images.append(image)
  nonTB_labels.append('test')


# ### Visualize First 40 Images from Data set

# In[26]:


# look at a random image for fun
def plot_images(images, title):
    nrows, ncols = 5, 8
    figsize = [10, 6]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=(1, 1, 1))

    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i])
        axi.set_axis_off()

    plt.suptitle(title, fontsize=24)
    plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
    plt.show()
plot_images(TB_images, 'Positive Tuberculosis Chest X-ray')
plot_images(nonTB_images, 'Negative Tuberculosis Chest X-ray')


# ### **Normalization**
# #### Model takes images in the form of array of pixels. Hence convert into array and *normalize*

# In[7]:


# Convert to array and Normalize to interval of [0,1]
TB_images = np.array(TB_images) / 255
nonTB_images = np.array(nonTB_images) / 255


# ### **Train Test Split**

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

# split into training and testing
TB_x_train, TB_x_test, TB_y_train, TB_y_test = train_test_split(
    TB_images, TB_labels, test_size=0.2)
nonTB_x_train, nonTB_x_test, nonTB_y_train, nonTB_y_test = train_test_split(
    nonTB_images, nonTB_labels, test_size=0.2)


X_train = np.concatenate((nonTB_x_train, TB_x_train), axis=0)
X_test = np.concatenate((nonTB_x_test, TB_x_test), axis=0)
y_train = np.concatenate((nonTB_y_train, TB_y_train), axis=0)
y_test = np.concatenate((nonTB_y_test, TB_y_test), axis=0)

# make labels into categories - either 0 or 1
y_train = LabelBinarizer().fit_transform(y_train)
y_train = to_categorical(y_train)

y_test = LabelBinarizer().fit_transform(y_test)
y_test = to_categorical(y_test)


# ### Visualize a few images from Training and Test sets

# In[27]:


plot_images(TB_x_train, 'X_train')
plot_images(TB_x_test, 'X_test')
# y_train and y_test contain class lables 0 and 1 representing COVID and NonCOVID for X_train and X_test


# ### **Building and Visualizing model**

# In[10]:


vggModel = VGG19(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs = vggModel.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(2, activation="softmax")(outputs)

model = Model(inputs=vggModel.input, outputs=outputs)

for layer in vggModel.layers:
    layer.trainable = False

model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)


# In[11]:


model.summary()


# ### **Image Augmentation**
# To train on images at different positions, angles, flips, etc

# In[12]:


train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# ### **Training the model**

# In[13]:


history = model.fit(train_aug.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / 32,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=100)


# In[14]:


model.save('vgg_chest.h5')


# In[15]:


model.save_weights('vggweights_chest.hdf5')


# In[16]:


model = load_model('vgg_chest.h5')


# ### **Making Predicions**

# In[17]:


y_pred = model.predict(X_test, batch_size=batch_size)


# ### Visulaizing First 10 predictions

# In[18]:


prediction=y_pred[0:10]
for index, probability in enumerate(prediction):
  if probability[1] > 0.5:
        plt.title('%.2f' % (probability[1]*100) + '% COVID')
  else:
        plt.title('%.2f' % ((1-probability[1])*100) + '% NonCOVID')
  plt.imshow(X_test[index])
  plt.show()


# In[19]:


# Convert to Binary classes
y_pred_bin = np.argmax(y_pred, axis=1)
y_test_bin = np.argmax(y_test, axis=1)


# ### Plot ROC Curve

# In[20]:


fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_bin)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for our model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)


# ### Plot Confusion Matrix

# In[21]:


def plot_confusion_matrix(normalize):
  classes = ['Tuberculosis','NonTB']
  tick_marks = [0.5,1.5]
  cn = confusion_matrix(y_test_bin, y_pred_bin,normalize=normalize)
  sns.heatmap(cn,cmap='plasma',annot=True)
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

print('Confusion Matrix without Normalization')
plot_confusion_matrix(normalize=None)

print('Confusion Matrix with Normalized Values')
plot_confusion_matrix(normalize='true')


# ### **Classification Report**

# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test_bin, y_pred_bin))


# ### **Accuracy and Loss Plots**

# In[23]:


plt.figure(figsize=(10,10))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('vgg_chest_accuracy.png')
plt.show()


# In[24]:


plt.figure(figsize=(10,10))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('vgg_chest_loss.png')
plt.show()


# In[ ]:





# In[ ]:




