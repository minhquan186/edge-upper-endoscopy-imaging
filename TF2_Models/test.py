#!/home/mcn/anaconda3/envs/quanle/bin/python3

import tensorflow as tf

if __name__ == '__main__':
    path = '/home/mcn/MinhQuan_K63/edge-upper-endoscopy-imaging/TF2_Models/models/mbnetv2_WLI_'
    loaded = tf.keras.models.load_model(path)
    loaded.summary()