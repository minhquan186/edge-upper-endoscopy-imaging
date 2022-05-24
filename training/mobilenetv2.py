#!/home/mcn/anaconda3/envs/quan/bin/python
import imp
from time import strftime
import numpy as np
import cv2
import os
from datetime import date
import pickle
import tensorboard
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

TRAIN_IMG_SIZE = 128
class_names =   [ 
                '1_Hau_hong',
                '2_Thuc_quan',
                '3_Tam_vi',
                '4_Than_vi',
                '5_Phinh_vi',
                '6_Hang_vi',
                '7_Bo_cong_lon',
                '8_Bo_cong_nho',
                '9_Hanh_ta_trang',
                '10_Ta_trang'
                ]
lightType = ['FICE', 'LCI', 'BLI', 'WLI']
nb_classes = len(class_names)

# One hot encoding
def labelEncoder(label):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label)


    # onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder = OneHotEncoder(sparse=False,dtype=np.uint8)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    encoded_label = onehot_encoded

    return encoded_label

def loadimage(folder_path, subset, IMAGE_SIZE=TRAIN_IMG_SIZE):
    images = []
    labels = []
    for type in range(len(lightType)):
        for i in range(nb_classes):
            label = i
            fold = f'{folder_path}/{lightType[type]}/{subset}/{class_names[i]}'
            for file in tqdm(os.listdir(fold)):
                img_path = os.path.join(fold, file)
                image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
                # image = cv2.cvtColor(image, cv2.COLOR_2RGB)
                image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
                images.append(image)
                labels.append(label)
                # labels.append(class_names[i])

    images = np.array(images, dtype = 'int32')
    labels = np.array(labels)
    # labels = labelEncoder(labels)
    return images, labels

def _save_pkl(path, obj):
      with open(path, 'wb') as f:
        pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj

def train_mobilenetv2(train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls, batchs, epochs):
    now = date.today()
    tsb_logdir = 'training/log-'+now.strftime('%d%m%Y')
    if not os.path.exists(tsb_logdir):
          os.makedirs(tsb_logdir)

    base_model = MobileNetV2(   
                             include_top=False,weights="imagenet",
                             input_shape=(TRAIN_IMG_SIZE,TRAIN_IMG_SIZE,3),
                             classes=nb_classes
                            )
    model = keras.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(nb_classes,activation="softmax"))
    model.compile(
                    optimizer = "adam", 
                    loss = 'sparse_categorical_crossentropy' , 
                    metrics = ['accuracy']
                 )
    callback1 = ModelCheckpoint(
                                filepath='/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/checkpoints/best_model.h5', 
                                save_best_only=True, 
                                save_weights_only=False
                                )
    callback2 = TensorBoard(log_dir=tsb_logdir)
    callback3 = ReduceLROnPlateau(
                                    monitor = 'val_loss', 
                                    patience = 2, 
                                    verbose = 1, 
                                    factor = 0.5, 
                                    min_lr = 0.000001
                                 )
    callback4 = ModelCheckpoint(
                                filepath='/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/checkpoints/last_model.h5', 
                                save_best_only=False,
                                save_weights_only=False
                                )
    hist = model.fit(
                        train_imgs,
                        train_lbls,
                        batch_size = batchs, 
                        epochs = epochs, 
                        validation_data=(val_imgs, val_lbls), 
                        callbacks=[callback1, callback2, callback3, callback4]
                    )
    res = model.evaluate(test_imgs, test_lbls)
    print("Test Loss:\t", res[0])
    print("Test Accuracy:\t", res[1]*100, "%")

if __name__ == '__main__':

    path = '/home/mcn/DATA_CHUAN/1_Vi_tri_giai_phau/'

    # train_images, train_labels = loadimage(path, 'train')
    # _save_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/train_imgs', train_images)
    # _save_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/train_lbls', train_labels)
    
    # val_images, val_labels = loadimage(path, 'val')
    # _save_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/val_imgs', val_images)
    # _save_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/val_lbls', val_labels)
    
    # test_images, test_labels = loadimage(path, 'test')
    # _save_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/test_imgs', test_images)
    # _save_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/test_lbls', test_labels)
    
    train_images = _load_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/train_imgs')
    train_labels = _load_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/train_lbls')
    
    val_images = _load_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/val_imgs')
    val_labels = _load_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/val_lbls')
    
    test_images = _load_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/test_imgs')
    test_labels = _load_pkl('/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/pkl/test_lbls')    

    print("Training images: {}".format(train_images.shape))
    print("Training labels: {}".format(train_labels.shape))
    print("Validation images: {}".format(val_images.shape))
    print("Validation labels: {}".format(val_labels.shape))
    print("Test images: {}".format(test_images.shape))
    print("Test labels: {}".format(test_labels.shape))

    train_mobilenetv2(train_images, train_labels, val_images, val_labels, test_images, test_labels, 16, 50)
