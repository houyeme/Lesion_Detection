import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf


import os



from glob import glob
from PIL import Image as pil_image
from matplotlib.pyplot import imshow, imsave
from IPython.display import Image as Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.utils.np_utils import to_categorical
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras import regularizers, initializers
from keras.layers.advanced_activations import LeakyReLU, ReLU, Softmax
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetLarge
from keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16

#import the data 
df = pd.read_csv('Training_GroundTruth.csv')

conditions=[(df['MEL']==1),(df['NV']==1),(df['CL']==1),(df['UNK']==1)]
values=[2,1,3,0]
df['type']=np.select(conditions,values)

image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join('', '*', '*.jpg'))}
df['path'] = df['image'].map(image_path.get)
df=df.drop(columns=['NV', 'CL','UNK','MEL'])

print(df)
image_example = np.asarray(pil_image.open(df['path'][0]))
print(image_example)



print(df.isnull().sum(axis =0))
df=df.loc[df['path'].notnull()]
print(df.isnull().sum(axis =0))
df['im'] = df['path'].map(lambda x: np.asarray(pil_image.open(x).resize((120,120))))



lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'unk':'unknown',
    'cl':'cutaneous leishmaniasis'
    
   
}

lesion_classes_dict = {
    0:'unk',
    1:'nv',
    2:'mel',
    3:'cl'
    
   
}

df['cell_type'] = df['type'].map(lesion_classes_dict)
features = df.drop(['type'],axis=1)
target = df['type']
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(features,target,test_size=0.01)
x_train = np.asarray(X_TRAIN['im'].tolist())
x_test = np.asarray(X_TEST['im'].tolist())
train_mean = x_train.mean()
train_std = x_train.std()
test_mean = x_test.mean()
test_std = x_test.std()
x_train = (x_train-train_mean) / train_std
x_test = (x_test-test_mean) / test_std
y_train = to_categorical(Y_TRAIN,num_classes=4)
y_test = to_categorical(Y_TEST,num_classes=4)
X_train,X_val, Y_train,Y_val = train_test_split(x_train,y_train,test_size=0.15)
X_train  = X_train.reshape(X_train.shape[0],120,120,3)
x_test  = x_test.reshape(x_test.shape[0],120,120,3)
X_val  = X_val.reshape(X_val.shape[0],120,120,3)
#checkpoint
checkpoint_path = "training_1/cprestntTake3.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
def resnet50():
    height=120
    width=120
   
    
    resnet = ResNet50(include_top=False, input_shape=(height, width, 3))
    x = resnet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(4,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

    model = Model(inputs=resnet.input, outputs=predictions)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model=resnet50()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.0001, 
                                            min_lr=0.000001)
datagen = ImageDataGenerator(
        
        zoom_range = 0.1 # Randomly zoom image 
)


# Fit the model
epochs = 65
batch_size = 10
model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction,cp_callback])
model.save('saved_model/resnetModelTake3')
def plot_(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    f, [ax1, ax2] = plt.subplots(1,2, figsize=(15, 5))
    ax1.plot(range(len(acc)), acc, label="accuracy")
    ax1.plot(range(len(acc)), val_acc, label="val_acc")
    ax1.set_title("Training Accuracy vs Validation Accuracy")
    ax1.legend()
    

    ax2.plot(range(len(loss)), loss, label="loss")
    ax2.plot(range(len(loss)), val_loss, label="val_loss")
    ax2.set_title("Training Loss vs Validation Loss")
    ax2.legend()
    plt.savefig('restnetTake3.png')
    print('saved')

    
plot_(model.history)

