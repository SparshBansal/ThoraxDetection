import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import glob

import model
import input
import constants

from PIL import Image

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
        'class_11',
        'class_12',
        'class_13',
        'class_14',
    ])

    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, default_value=-1)
    tf.tables_initializer().run()

    return table


def preprocess_labels(labels, table):
    # define classes and a lookup table
    class_labels = labels[:, -1]
    idx_labels = table.lookup(class_labels)

    return idx_labels


def train_neural_network():
    tf.reset_default_graph()

    with tf.Session() as sess:

        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # initialize lookup table
        table = initialize_lookup_table()

        train_feature_filenames, train_label_filenames = get_filenames();

        with tf.name_scope('raw_inputs'):
            features, raw_labels = input.getFiles(train_feature_filenames, train_label_filenames)

        with tf.name_scope('processed_labels'):
            labels = preprocess_labels(raw_labels, table)

        output, test_output, test_features, test_labels = model.create_model(features, labels)

        with tf.name_scope('loss'):
            loss = model.get_loss(output, labels)

        with tf.name_scope('training_accuracy'):
            training_accuracy = model.compute_accuracy(output, labels)

        with tf.name_scope('dev_accuracy'):
            dev_accuracy = model.compute_accuracy(test_output, test_labels)

        train_step = model.get_optimizer(loss)
        training_fetches = [features,raw_labels,labels, output, loss, training_accuracy, train_step]

        # initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # add graph summary for tensorboard
        writer = tf.summary.FileWriter(constants.TENSORBOARD_DIR, sess.graph)

        # start queue runner for data loading
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # get dev features
        dev_features, dev_labels = sess.run([features, labels])
        # check if we received the labels correctly or not
        print dev_labels

        for epoch in range(1, constants.EPOCHS + 1):
            for batch in range(1, constants.NUM_BATCHES + 1):
                # train the model
                model_features,model_raw_labels,model_labels, model_output, model_loss, model_accuracy, _ = sess.run(training_fetches)
                print "Epoch {}/{} ; Batch {}/{} ; Accuracy {} ; Loss {}".format(epoch, constants.EPOCHS, batch,
                                                                                 constants.NUM_BATCHES, model_accuracy,
                                                                                 model_loss)
                print model_output  
                # evaluate the accuracy
                if (batch % constants.TEST_PERIOD == 0):
                    mdev_accuracy = sess.run(dev_accuracy,
                                             feed_dict={test_features: dev_features, test_labels: dev_labels})


train_neural_network()
