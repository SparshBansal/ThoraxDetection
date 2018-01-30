import tensorflow as tf
import numpy as np
import glob

import model
import input

train_dir = './data/train/downsampled/'
test_dir = './data/test/downsampled/'

def get_filenames():
    train_feature_filenames = glob.glob(train_dir + "*.png")
    train_feature_filenames.sort()

    train_label_filenames = ["./data/train.csv"]
    return train_feature_filenames, train_label_filenames

def train_neural_network():

    train_feature_filenames, train_label_filenames= get_filenames();

    feature_batch , label_batch = input.getFiles(train_feature_filenames , train_label_filenames)
    output = model.conv_nn(feature_batch)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # start queue runner for data loading
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        model_output = sess.run(output)
        loss  =
        print model_output

train_neural_network()
