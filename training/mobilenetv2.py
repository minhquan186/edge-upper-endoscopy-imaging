#!/home/mcn/anaconda3/envs/quan/bin/python

# Import libs
import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
import tensorflow as tf
# import argparse
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder

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
lightType = ['BLI', 'FICE', 'LCI', 'WLI']
nb_classes = len(class_names)

def load_img(data_path, subset, img_size):
    images = []
    labels = []
    cnt = 0
    for light in range(len(os.listdir(data_path))):
        for item in class_names:
            path = f'{data_path}/{lightType}'
        