import tensorflow as tf
import numpy as np
import glob
import os

def getFiles(filenames, labelnames):
    featureinput_producer = tf.train.string_input_producer(filenames,shuffle=False)
    labelinput_producer = tf.train.string_input_producer(labelnames,shuffle=False)

    feature_reader = tf.WholeFileReader()
    label_reader = tf.TextLineReader()

    _,feature_value = feature_reader.read(featureinput_producer)
    _,label_value = label_reader.read(labelinput_producer)

    image = tf.image.decode_png(feature_value,3)
    image = tf.reshape(image,[512,512,3])

    defaults = [["a"],[1],['M'],[1],['noimage'],['noclass']]

    col1,col2,col3,col4,col5,col6 = tf.decode_csv(label_value,record_defaults=defaults)
    label = tf.stack([col1,col3,col5,col6])

    feature_batch , label_batch = tf.train.batch([image,label],batch_size=5)

    return tf.cast(feature_batch, tf.float32),label_batch

