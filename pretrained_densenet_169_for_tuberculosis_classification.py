# -*- coding: utf-8 -*-

'''
from google.colab import drive
drive.mount('/content/drive')
'''
#!pip install scikit-plot

# In[]

# import all libraries
# Train/Test Libraries
import json
from pathlib import Path
import cv2
import os
import numpy as np
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scikitplot
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

# In[]
# %cd '/content/drive/MyDrive/FinalProject/Medical Image Investigation/TB Impl paper/Tuberculosis-master/Classification'

IMG_SIZE = 224

train_x = np.load('/home/taherizade/Sajid Ali Final Project/Project Codes/Grad Cam code/TB impl/Tuberculosis-master/Numpy files/train_images.npy')
train_y = np.load('/home/taherizade/Sajid Ali Final Project/Project Codes/Grad Cam code/TB impl/Tuberculosis-master/Numpy files/train_labels.npy')
test_x = np.load('/home/taherizade/Sajid Ali Final Project/Project Codes/Grad Cam code/TB impl/Tuberculosis-master/Numpy files/valid_images.npy')
test_y = np.load('/home/taherizade/Sajid Ali Final Project/Project Codes/Grad Cam code/TB impl/Tuberculosis-master/Numpy files/valid_labels.npy')

print('Training Images: {} | Test Images: {}'.format(train_x.shape, test_x.shape))
print('Training Labels: {} | Test Labels: {}'.format(train_y.shape, test_y.shape))

# Data Normalization

print('Train: {} , {} | Test: {} , {}'.format(train_x.min(), train_x.max(), test_x.min(), test_x.max()))

train_x/=255.0
test_x/=255.0

print('Train: {} , {} | Test: {} , {}'.format(train_x.min(), train_x.max(), test_x.min(), test_x.max()))

# Class Mapping 
print('0:Healthy | 1:Tuberculosis')

# Distribution of images in each class for Training-set
print(Counter(train_y))

# Distribution of images in each class for Test-set
print(Counter(test_y))

# Make Labels Categorical
train_y_oneHot = tf.one_hot(train_y, depth=2) 
test_y_oneHot = tf.one_hot(test_y, depth=2)

print('Training Labels: {} | Test Labels: {}'.format(train_y_oneHot.shape, test_y_oneHot.shape))

# initialize the training data augmentation object
trainAug = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range = 0.15,
                                                          width_shift_range = 0.15,
                                                          rotation_range = 10,
                                                          shear_range = 0.1,
                                                          fill_mode = 'nearest',
                                                          zoom_range=0.2
                                                          )
def DenseNet169_Model():
  # load the DenseNet169 network, ensuring the head FC layer sets are left off
  baseModel = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling=None)
  # construct the head of the model that will be placed on top of the the base model
  output = baseModel.output
  output = tf.keras.layers.GlobalAveragePooling2D()(output)
  output = tf.keras.layers.Dense(1024, activation="relu")(output)
  output = tf.keras.layers.Dropout(0.15)(output)
  output = tf.keras.layers.Dense(512, activation="relu")(output)
  output = tf.keras.layers.Dropout(0.15)(output)
  output = tf.keras.layers.Dense(2, activation="softmax")(output)
  # place the head FC model on top of the base model (this will become the actual model we will train)
  model = tf.keras.Model(inputs=baseModel.input, outputs=output)
  # loop over all layers in the base model and freeze them so they will not be updated during the first training process
  for layer in baseModel.layers:
    layer.trainable = False
  return model

model = DenseNet169_Model()
# compile our model
print("[INFO] compiling model...")
# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 0.001
EPOCHS = 100
BATCHSIZE = 32 
optimizer = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])
print(model.summary())

modelPath = '/home/taherizade/Sajid Ali Final Project/Project Codes/DensNEt code and results/save models'
if not os.path.exists(modelPath):
  os.makedirs(modelPath)
  print('Model Directory Created')
else:
  print('Model Directory Already Exists')

reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.8, patience=10, verbose=1, mode='auto',
                                                      min_delta=0.0001, cooldown=5, min_lr=0.0001)
early = tf.keras.callbacks.EarlyStopping(monitor="val_categorical_accuracy", mode="auto", patience=10)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(modelPath+'/denseNet169-best-model.h5', monitor='val_categorical_accuracy',
                                                      verbose=1, save_best_only=True, mode='auto')

STEP_TRAIN = len(train_x) // BATCHSIZE


modelHistory = model.fit(trainAug.flow(train_x, train_y_oneHot, batch_size=BATCHSIZE), steps_per_epoch=STEP_TRAIN, 
                         validation_data= (test_x, test_y_oneHot), epochs=EPOCHS, verbose=1, callbacks=[model_checkpoint, reduceLROnPlat])

tf.keras.models.save_model(model, modelPath+'/DenseNet169-model.h5', overwrite=True, include_optimizer=True, save_format=None,
                           signatures=None, options=None)

# Evaluate the Best Saved Model
model = tf.keras.models.load_model('/home/taherizade/Sajid Ali Final Project/Project Codes/Grad Cam code/TB impl/Tuberculosis-master/Classification/saved Models/Classification/Pretrained DenseNet169/denseNet169-best-model.h5')
loss, accuracy, auc= model.evaluate(x=test_x, y=test_y_oneHot, batch_size=32, verbose=1)
print('Model Accuracy: {:0.2f} | Model Loss: {:0.4f} | Model AUC: {:.02f}'.format(accuracy, loss, auc))

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Purples')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('GroundTruths')
    plt.xlabel('Predictions \n Model Accuracy={:0.2f}% | Model Error={:0.2f}%'.format(accuracy*100, misclass*100))
    plt.savefig('/home/taherizade/Sajid Ali Final Project/Project Codes/Grad Cam code/TB impl/Tuberculosis-master/ReadMe Images/DenseNet169-cm.png', bbox_inches = "tight")
    plt.show()


predictions = model.predict(x=test_x, batch_size=32)
predictions = tf.keras.backend.argmax(predictions, axis=-1)

test_y = tf.keras.backend.argmax(test_y_oneHot, axis=-1)
cm = confusion_matrix(test_y, predictions)
classes = ['Healthy', 'Pulmonary TB']
plot_confusion_matrix(cm=cm, normalize = False, target_names = classes, title= "Confusion Matrix (Pretrained DenseNet121)")


# In[]
y_true = [0]
y_pred = [1]
target_names = ['NonTB', 'Tuberculosis']
print(classification_report(y_true=test_y,
      y_pred=predictions, target_names=target_names))


# In[]
#Plot ROC Curve with Library
predictions = model.predict(x=test_x, batch_size=32)
# One can define colormap here
# cmap = mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
scikitplot.metrics.plot_roc(y_true=test_y, y_probas=predictions, title='ROC Curves (Pretrained DenseNet169)', plot_micro=False, plot_macro=False,
                            classes_to_plot=None, ax=None, figsize=(6, 4), cmap='Dark2', title_fontsize='large', text_fontsize='medium')
plt.savefig('/home/taherizade/Sajid Ali Final Project/Project Codes/Grad Cam code/TB impl/Tuberculosis-master/ReadMe Images/DenseNet169-roc.png', bbox_inches = "tight")
plt.show()



# In[]
# Commented out IPython magic to ensure Python compatibility.
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %cd '/content/drive/MyDrive/FinalProject/Medical Image Investigation/TB Impl paper/Tuberculosis-master/Classification'

print(tf.__version__)
validDir =  '/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/chest_xray_TBNew_TestDataset/validation'
LAYER_NAME = 'relu'
model = tf.keras.models.load_model('/home/taherizade/Sajid Ali Final Project/Project Codes/DensNEt code and results/save models/DenseNet169-model.h5')
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])
#grad_model = tf.keras.models.Model([model.inputs], [model.get_layer('block5_conv3').output, model.output])# new line added from stack


def GetHeatMap(img):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, np.argmax(predictions)]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    heatmap = cv2.addWeighted(np.uint8(255*img), 0.6, cam, 0.4, 0)
    return heatmap

imgSize = 224
channels = 3
imgNames = ['tb0021.png', 'tb0124.png', 'tb0054.png', 'tb0378.png', 'tb0402.png', 'tb0527.png']
images = []
for imgName in imgNames:
    imgPath = os.path.join(validDir, '1', imgName)
    img = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(imgPath, target_size=(imgSize, imgSize)), dtype='float32')/255.0
    images.append(img)

images = np.array(images).reshape(-1, imgSize, imgSize, channels)
print(images.shape)

rows = 3
cols = 4
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 8), subplot_kw={'xticks': [], 'yticks': []})
img_count =1
for r in range(rows):
    # plot image
    ax[r][0].imshow(images[img_count, :, :, :])
    ax[r][0].set_title(imgNames[img_count], fontsize=12)

    # plot heatmap
    ax[r][1].imshow(images[img_count, :, :, :])
    ax[r][1].imshow(GetHeatMap(images[img_count, :, :, :]))
    ax[r][1].set_title(imgNames[img_count]+ '(Heatmap)', fontsize=12)

    img_count+=1
    # plottuberculosis image
    ax[r][2].imshow(images[img_count, :, :, :])
    ax[r][2].set_title(imgNames[img_count], fontsize=12)

    # plot heatmap
    ax[r][3].imshow(images[img_count, :, :, :])
    ax[r][3].imshow(GetHeatMap(images[img_count, :, :, :]))
    ax[r][3].set_title(imgNames[img_count]+ '(Heatmap)', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0)
#fig.savefig('/home/taherizade/Sajid Ali Final Project/Project Codes/Grad Cam code/TB impl/Tuberculosis-master/ReadMe Images/denseNet169-visualization.png')
plt.show()
plt.close()


# In[]:
    
# add loding json

source_path = Path(
    '/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/imgs/annotations/json')


f1 = open(source_path / 'all_train.json')

j = json.load(f1)

image_names = [(x['id'], x['file_name']) for x in j['images']]

annotations = {}

for x in j['annotations']:
    annotations[x['id']] = x['bbox']


##    


zzz = '/home/taherizade/Sajid Ali Final Project/Project Codes/DensNEt code and results/save models'

validDir = '/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/imgs'
LAYER_NAME = 'relu'
model = tf.keras.models.load_model(
    f'{zzz}/DenseNet169-model.h5')
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])


p = Path('/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/chest_xray_TBNew_TestDataset/validation')

p = p/ '1'


list_of_all_files = sorted(list(p.glob('*.png')))


# imgNames = ['tb0021.png', 'tb0248.png',
#             'tb0108.png', 'tb0124.png', 'tb0146.png']

imgNames = [x.name for x in list_of_all_files]


c_i = 0


has_overlap = 0

accuracy_list = []


for imgName in imgNames:
    
    ### ramin added
    
    # if imgName != 'tb1034.png':
    #     continue
    
    bbox = None
    name_1 = None
    for id_1, name in image_names:
        if imgName in name:
            bbox = annotations[id_1]
            name_1 = name
            break
    
    if not bbox:
        continue
    else:
        c_i += 1
    
    if c_i >= 100:
        break;
    
    ### ramin added

    IMAGE_PATH = os.path.join(validDir, '1', imgName)
    img = tf.keras.preprocessing.image.load_img(
        IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img, dtype='uint8')
    img_array = img/255.0
    predictions = model.predict(np.expand_dims(img_array, axis=0))
    if np.argmax(predictions) == 1:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([img_array]))
            loss = predictions[:, np.argmax(predictions)]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = tf.cast(output > 0, 'float32') * \
            tf.cast(grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam.numpy(), (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        
        heatmap = heatmap > heatmap.max() * 0.75

        cam_gray = heatmap

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

        img_2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        output_image = cv2.addWeighted(img, 0.8, cam, 0.5, 0)

        output_image_2 = img_2 * 0.8 + cam_gray*255 * 0.5

        rows = 1
        cols = 3
        fig, ax = plt.subplots(nrows=rows, ncols=cols,
                               figsize=(9.5, 10.5), squeeze=False)
        for r in range(rows):
            # plot image
            ax[r][0].imshow(img)
            ax[r][0].set_title(imgName, fontsize=15)
            ax[r][0].axis('off')

            # plot image
            ax[r][1].imshow(cam_gray, cmap='gray')
            ax[r][1].set_title('Binarized output Image', fontsize=15)
            ax[r][1].axis('off')

            ### ramin added
            output_image_3 = np.zeros(
                [output_image_2.shape[0], output_image_2.shape[1], 3], dtype=output_image_2.dtype)

            output_image_3[:, :, 0] = output_image_2
            output_image_3[:, :, 1] = output_image_2
            output_image_3[:, :, 2] = output_image_2

            output_image_3 = (output_image_3 - output_image_3.min()) / \
                (output_image_3.max() - output_image_3.min())

            # plot heatmap
            if bbox:

                img_k1 = cv2.imread(f'{validDir}/1/{imgName}')

                x_scale = 224/img_k1.shape[1]
                y_scale = 224/img_k1.shape[0]

                bbox = list(map(int, bbox))

                x1 = int(bbox[0] * x_scale)
                y1 = int(bbox[1] * y_scale)
                x2 = int((bbox[0]+bbox[2])*x_scale)
                y2 = int((bbox[1]+bbox[3])*y_scale)

                cv2.rectangle(output_image_3, (x1, y1),
                              (x2, y2),
                              (1, 0, 0), 2)
                
                
                raster_boundary = np.ones_like(heatmap)
                
                
                y = np.arange(raster_boundary.shape[0])
                x = np.arange(raster_boundary.shape[1])
                
                xv, yv = np.meshgrid(x,y)
                
                
                raster_boundary[xv < x1] = 0
                raster_boundary[xv > x2] = 0
                raster_boundary[yv < y1] = 0
                raster_boundary[yv > y2] = 0
                
                
                idk = raster_boundary * 1 + heatmap * 1
                
                to_show_idk = idk > 1
                all_points = idk > 0
                
                if to_show_idk.any() == True:
                    has_overlap += 1
                    print(f'{c_i}: {imgName}')
                
                # raster_boundary[bbox] = 
                
                accuracy_list.append(to_show_idk.sum()/all_points.sum())

            ax[r][2].imshow(output_image_3)
            
            ### ramin added
            
            ax[r][2].set_title('Thresholded heatmap image', fontsize=15)
            ax[r][2].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.close()
        
print(f'has_overlap_percentages: {(has_overlap/c_i)*100}%')
print(f'accuracy: {np.mean(accuracy_list) * 100}%')

# In[]:
    
    
# add loding json

source_path = Path(
    '/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/imgs/annotations/json')


f1 = open(source_path / 'all_train.json')

j = json.load(f1)

image_names = [(x['id'], x['file_name']) for x in j['images']]

annotations = {}

for x in j['annotations']:
    annotations[x['id']] = x['bbox']


##    


zzz = '/home/taherizade/Sajid Ali Final Project/Project Codes/DensNEt code and results/save models'

validDir = '/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/imgs'
LAYER_NAME = 'relu'
model = tf.keras.models.load_model(
    f'{zzz}/DenseNet169-model.h5')
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])    

p = Path('/home/taherizade/Sajid Ali Final Project/Project Codes/MainCode MEdi Image Analysis/chest_xray_TBNew_TestDataset/validation')

p = p/ '1'


list_of_all_files = sorted(list(p.glob('*.png')))


# imgNames = ['tb0021.png', 'tb0248.png',
#             'tb0108.png', 'tb0124.png', 'tb0146.png']

imgNames = [x.name for x in list_of_all_files]


c_i = 0


has_overlap = 0

accuracy_list = []

predictions_list = []
before_list = []



for imgName in imgNames:
    
    ### ramin added
    
    # if imgName != 'tb1034.png':
    #     continue
    
    bbox = None
    name_1 = None
    for id_1, name in image_names:
        if imgName in name:
            bbox = annotations[id_1]
            name_1 = name
            break
    
    if bbox is None:
        continue
    else:
        c_i += 1
    
    if c_i >= 100:
        break;
    
    ### ramin added

    IMAGE_PATH = os.path.join(validDir, '1', imgName)
    img = tf.keras.preprocessing.image.load_img(
        IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img, dtype='uint8')
    
    # suspecious :)
    img_array = img/255.0
    
    predictions = model.predict(np.expand_dims(img_array, axis=0))
    before_list.append(np.argmax(predictions) == 1)
    
    # slicing image
    
    
    slice_t = 4
    
    x_s = img_array.shape[0] // slice_t
    y_s = img_array.shape[1] // slice_t
    
    
    predictions_list.append([])
    f1 = 0
    for i_k in range(0,img_array.shape[0] -img_array.shape[0] % slice_t,x_s):
        for j_k in range(0,img_array.shape[1] - img_array.shape[1] % slice_t,y_s):
            img_k = np.zeros_like(img)
            img_k[i_k:i_k + x_s,j_k:j_k+y_s] = img[i_k:i_k+x_s,j_k:j_k+y_s]
            
            img_array = img_k/255.0
    
            predictions = model.predict(np.expand_dims(img_array, axis=0))

            predictions_list[-1].append(np.argmax(predictions) == 1)
            
                
            if True:
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(np.array([img_array]))
                    loss = predictions[:, np.argmax(predictions)]
        
                output = conv_outputs[0]
                grads = tape.gradient(loss, conv_outputs)[0]
        
                gate_f = tf.cast(output > 0, 'float32')
                gate_r = tf.cast(grads > 0, 'float32')
                guided_grads = tf.cast(output > 0, 'float32') * \
                    tf.cast(grads > 0, 'float32') * grads
        
                weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        
                cam = np.ones(output.shape[0: 2], dtype=np.float32)
        
                for i, w in enumerate(weights):
                    cam += w * output[:, :, i]
        
                cam = cv2.resize(cam.numpy(), (224, 224))
                cam = np.maximum(cam, 0)
                heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        
                heatmap = heatmap > heatmap.max() * 0.75
        
                cam_gray = heatmap
        
                cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
                cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        
                img_2 = cv2.cvtColor(img_k, cv2.COLOR_RGB2GRAY)
        
                output_image = cv2.addWeighted(img_k, 0.8, cam, 0.5, 0)
        
                output_image_2 = img_2 * 0.8 + cam_gray*255 * 0.5
        
                rows = 1
                cols = 3
                fig, ax = plt.subplots(nrows=rows, ncols=cols,
                                       figsize=(9.5, 10.5), squeeze=False)
                for r in range(rows):
                    # plot image
                    ax[r][0].imshow(img_k)
                    ax[r][0].set_title(imgName, fontsize=15)
                    ax[r][0].axis('off')
        
                    # plot image
                    ax[r][1].imshow(cam_gray, cmap='gray')
                    ax[r][1].set_title('Filtermap', fontsize=15)
                    ax[r][1].axis('off')
        
                    ### ramin added
                    output_image_3 = np.zeros(
                        [output_image_2.shape[0], output_image_2.shape[1], 3], dtype=output_image_2.dtype)
        
                    output_image_3[:, :, 0] = output_image_2
                    output_image_3[:, :, 1] = output_image_2
                    output_image_3[:, :, 2] = output_image_2
        
                    output_image_3 = (output_image_3 - output_image_3.min()) / \
                        (output_image_3.max() - output_image_3.min())
        
                    # plot heatmap
                    if bbox:
        
                        img_k1 = cv2.imread(f'{validDir}/1/{imgName}')
        
                        x_scale = 224/img_k1.shape[1]
                        y_scale = 224/img_k1.shape[0]
        
                        bbox = list(map(int, bbox))
        
                        x1 = int(bbox[0] * x_scale)
                        y1 = int(bbox[1] * y_scale)
                        x2 = int((bbox[0]+bbox[2])*x_scale)
                        y2 = int((bbox[1]+bbox[3])*y_scale)
        
                        cv2.rectangle(output_image_3, (x1, y1),
                                      (x2, y2),
                                      (1, 0, 0), 2)
                        
                        
                        raster_boundary = np.ones_like(heatmap)
                        
                        
                        y = np.arange(raster_boundary.shape[0])
                        x = np.arange(raster_boundary.shape[1])
                        
                        xv, yv = np.meshgrid(x,y)
                        
                        
                        raster_boundary[xv < x1] = 0
                        raster_boundary[xv > x2] = 0
                        raster_boundary[yv < y1] = 0
                        raster_boundary[yv > y2] = 0
                        
                        
                        idk = raster_boundary * 1 + heatmap * 1
                        
                        to_show_idk = idk > 1
                        all_points = idk > 0
                        
                        if to_show_idk.any() == True:
                            has_overlap += 1
                            print(f'{c_i}_{f1}: {imgName}')
                        
                        
                        accuracy_list.append(to_show_idk.sum()/all_points.sum())
                        
                        # raster_boundary[bbox] = 
        
                    ax[r][2].imshow(output_image_3)
                    
                    ### ramin added
                    
                    ax[r][2].set_title('Annotation and Prediction', fontsize=15)
                    ax[r][2].axis('off')
        
                plt.tight_layout()
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.show()
                plt.close()
                f1+=1
        
print(f'{predictions_list}')      
print(f'{before_list}') 
# print(f'has_overlap_percentages: {(has_overlap/c_i)*100}%')
# print(f'accuracy: {np.mean(accuracy_list) * 100}%')
