import tensorflow as tf
import numpy as np
import glob

import model
import input
import constants


def get_filenames():
    train_feature_filenames = glob.glob(constants.FEATURE_DIR + "*.png")
    train_feature_filenames.sort()
    train_label_filenames = [constants.LABEL_DIR + "train.csv"]
    return train_feature_filenames, train_label_filenames

def initialize_lookup_table():
    mapping_strings = tf.constant([
        'class_1',
        'class_2',
        'class_3',
        'class_4',
        'class_5',
        'class_6',
        'class_7',
        'class_8',
        'class_9',
        'class_10',
        'class_12',
        'class_13',
        'class_14',
        ])

    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, default_value=-1)
    tf.tables_initializer().run()

    return table


def preprocess_labels(labels, table):
    # define classes and a lookup table
    class_labels = labels[:,-1]
    idx_labels = table.lookup(class_labels)

    return idx_labels


def train_neural_network():

    tf.reset_default_graph()
    
    with tf.Session() as sess:

        # initialize lookup table
        table = initialize_lookup_table()

        train_feature_filenames, train_label_filenames= get_filenames();
        features , labels = input.getFiles(train_feature_filenames , train_label_filenames)
    
        labels = preprocess_labels(labels, table)

        #output = model.conv_nn(features)
        #loss = model.get_loss(output, labels)
        #train_step = model.get_optimizer(loss)

        fetches = [features , labels]

        sess.run(tf.global_variables_initializer())
        
        # start queue runner for data loading
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # run the graph
        f , l = sess.run(fetches)

        print l

train_neural_network()
