#!/home/mcn/anaconda3/envs/quan/bin/python
import numpy as np
import cv2
import os
from datetime import date
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
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
from sklearn.metrics import classification_report, confusion_matrix

'''
    Disable TF logging
'''
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(3)

'''
    Parameters
'''
lightType = 'WLI'
augmented_technique = ['BAC', 'aff']

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
folder_names = [
                'Vung hau hong', 
                'Thuc quan',
                'Tam vi',
                'Than vi',
                'Phinh vi', 
                'Hang vi',
                'Bo cong lon',
                'Bo cong nho',
                'Hanh ta trang',
                'Ta trang'
                ]
nb_classes = len(class_names)

# One hot encoding
def labelEncoder(label):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label)

    onehot_encoder = OneHotEncoder(sparse=False,dtype=np.uint8)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    encoded_label = onehot_encoded

    return encoded_label

def loadimage(datapath, subset, IMAGE_SIZE=TRAIN_IMG_SIZE):
    """
    Args:
        datapath[array]: path to the dataset
        subset[string]: path to subset (train, test, val)
    """
    images = []
    labels = []
    for folder_path in datapath:
        if (subset == 'val' or subset =='test'):
            for i in range(nb_classes):
                label = i
                fold = f'{folder_path}/{subset}/{class_names[i]}'
                for file in tqdm(os.listdir(fold)):
                    img_path = os.path.join(fold, file)
                    image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
                    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
                    images.append(image)
                    labels.append(label)
        elif subset == 'train':
            for i in range(nb_classes):
                label = i
                # fold = f'{folder_path}/{folder_names[i]}'
                fold = f'{folder_path}/{subset}/{class_names[i]}'
                for file in tqdm(os.listdir(fold)):
                    img_path = os.path.join(fold, file)
                    image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
                    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
                    images.append(image)
                    labels.append(label)
        else:
            print(f'Invalid subset: %s' % subset)
        
    images = np.array(images, dtype = 'int32')
    labels = np.array(labels, dtype = 'int8')
    # labels = labelEncoder(labels)
    return images, labels

def calculate_sen_spec(cm):
    size = len(cm)
    TP, F, TN, TF, f1score, total = 0, 0, 0, 0, 0, 0
    total += sum([sum([cm[i][j] for i in range(size)]) for j in range(size)])
    TP += sum([cm[i][i] for i in range(size)])
    F = total - TP
    TN = (size-2)*total+TP
    sen = TP/(TP+F)
    spec = TN/(TN+F)
    acc = (TP+TN)/(size*total)
    pre = TP/(TP+F)
    f1score = 2*TP/(2*TP+2*F)

    return sen, spec, acc, pre, f1score

def plot_confusion_matrix (cm, cm_name):
    cm_savedir = '/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/TF2_Models/res/'+cm_name
    plt.figure(figsize = (10,10))
    cm_graph = sns.heatmap(
        cm, 
        cmap = 'Blues', 
        linecolor = 'black', 
        linewidth = 1, 
        annot = True, 
        fmt = '.2%', 
        xticklabels = class_names, 
        yticklabels = class_names)
    cm_fig = cm_graph.get_figure()
    cm_fig.savefig(cm_savedir)

def train_mobilenetv2(train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls, batchs, epochs):
    log_dir = '/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/tsb_logs'
    # tsb_logdir =f'{log_dir}/log_{lightType}_{augmented_technique}'
    tsb_logdir =f'{log_dir}/log_AllLightType'
    if not os.path.exists(tsb_logdir):
          os.makedirs(tsb_logdir)
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
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
    best_ckpt = ModelCheckpoint(
                                filepath='/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/training/checkpoints/best_model.h5', 
                                save_best_only=True, 
                                save_weights_only=False,
                                verbose=1
                                )
    tsb = TensorBoard(log_dir=tsb_logdir)
    reduce_lr = ReduceLROnPlateau(
                                    monitor = 'val_loss', 
                                    patience = 2, 
                                    verbose = 1, 
                                    factor = 0.5, 
                                    min_lr = 0.000001
                                 )
    last_ckpt = ModelCheckpoint(
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
                        callbacks=[best_ckpt, tsb, reduce_lr, last_ckpt]
                    )
    # model.save(f'/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/TF2_Models/models/mbnetv2_{lightType}_{augmented_technique}')
    model.save(f'/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/TF2_Models/models/mbnetv2_AllLightType')
    res = model.evaluate(test_imgs, test_lbls)
    print("Test Loss:\t", res[0])
    print("Test Accuracy:\t", res[1]*100, "%")
    predictions = model.predict(test_images)
    predictions = np.argmax(predictions,axis=1)
    print(classification_report (
                                test_lbls,
                                predictions,
                                target_names = class_names
                                )
          )

    cm = confusion_matrix(test_labels, predictions)
    calculate_sen_spec(cm)
    print(cm)
    cm = pd.DataFrame(cm, index = folder_names, columns = folder_names)
    plot_confusion_matrix(cm, f'{lightType}_{augmented_technique}_og.pdf')
    # plot_confusion_matrix(cm, f'{lightType}_og.pdf')

def load_trained_model(path, test_imgs, test_lbls):
    loaded = tf.keras.models.load_model(path)
    res = loaded.evaluate(test_imgs, test_lbls)
    print("Test Loss:\t", res[0])
    print("Test Accuracy:\t", res[1]*100, "%")
    predictions = loaded.predict(test_imgs)
    predictions = np.argmax(predictions,axis=1)
    print(classification_report (
                                test_lbls,
                                predictions,
                                target_names = class_names
                                )
          )

    cm = confusion_matrix(test_labels, predictions)
    calculate_sen_spec(cm)
    print(cm)
    cm = pd.DataFrame(cm, index = folder_names, columns = folder_names)
    plot_confusion_matrix(cm, f'{lightType}_{augmented_technique}_loaded.pdf')

if __name__ == '__main__':
    env_cuda = 'CUDA_VISIBLE_DEVICES'
    if env_cuda in os.environ:
        print(f'GPU\'s ID in use: {env_cuda} = {os.environ[env_cuda]}')
    else:
        print('GPU is not specified in the environment. Using default value...')
    train_path = f'/home/mcn/MinhQuan_K63/Vi_tri_giai_phau_all'
    # train_path   = f'/home/mcn/DATA_CHUAN/1_Vi_tri_giai_phau/{lightType}'
    # train_path1 = f'/home/mcn/Tuan-K63/Retrain5000/{lightType}/augment_{augmented_technique[0]}'
    # train_path2 = f'/home/mcn/Tuan-K63/Retrain5000/{lightType}/augment_{augmented_technique[1]}'
    test_path   = f'/home/mcn/DATA_CHUAN/1_Vi_tri_giai_phau/{lightType}'

    # train_images, train_labels = loadimage([train_path1, train_path2], 'train')
    train_images, train_labels = loadimage([train_path], 'train')

    val_images, val_labels = loadimage([train_path], 'val')

    test_images, test_labels = loadimage([train_path], 'test')
  
    print("Training images: {}".format(train_images.shape))
    print("Training labels: {}".format(train_labels.shape))
    print("Validation images: {}".format(val_images.shape))
    print("Validation labels: {}".format(val_labels.shape))
    print("Test images: {}".format(test_images.shape))
    print("Test labels: {}".format(test_labels.shape))

    train_mobilenetv2 (
                        train_images, train_labels,
                        val_images, val_labels,
                        test_images, test_labels,
                        batchs=32,
                        epochs=50
                      )
    # model_path = f'/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/TF2_Models/models/mbnetv2_{lightType}_{augmented_technique}'
    # load_trained_model(model_path, test_images, test_labels)