import tensorflow as tf
import constants


class Model:

    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]

    def get_output(self):
        return self.outputs[-1]

    def get_num_layers(self):
        return len(self.outputs);

    def get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        return '%sL%03d' % (self.name, layer + 1)

    def get_num_inputs(self):
        return self.get_output().get_shape()[-1];

    def add_conv2d(self, kernel_size=5, output_channels=32, stride=4, stddev=1):
        assert len(
            self.get_output().get_shape()) == 4 and "Previous layer must be 4 dimensional (batch, width, height, channels)"

        with tf.variable_scope(self.get_layer_str()):
            input_channels = self.get_num_inputs()

            # make variables for weights and biases 
            weight = tf.get_variable('weight', shape=[kernel_size, kernel_size, input_channels, output_channels])
            bias = tf.get_variable('bias', initializer=tf.constant(0.0, shape=[output_channels]))

            out = tf.nn.conv2d(self.get_output(), weight, strides=[1, stride, stride, 1], padding='SAME')
            out = tf.nn.bias_add(out, bias)

        self.outputs.append(out)

    def add_relu(self):
        with tf.variable_scope(self.get_layer_str()):
            out = tf.nn.relu(self.get_output())
        self.outputs.append(out)

    # add fully connected layer
    def add_fc(self, output_units):
        inputs = tf.contrib.layers.flatten(self.get_output());
        with tf.variable_scope(self.get_layer_str()):
            out = tf.contrib.layers.fully_connected(inputs, output_units)

        self.outputs.append(out)


def conv_nn(features):
    model = Model('cnn', features)

    # add layers accordingly
    model.add_conv2d(5, 32)
    model.add_relu()
    model.add_conv2d(7, 64)
    model.add_relu()
    model.add_conv2d(7, 128)
    model.add_relu()

    model.add_fc(20)
    model.add_fc(14)

    return model.get_output()


def create_model(features, labels):
    _height = features.get_shape()[1]
    _width = features.get_shape()[2]
    _channel = features.get_shape()[3]

    test_features = tf.placeholder(tf.float32, [constants.BATCH_SIZE, _height, _width, _channel])
    test_labels = tf.placeholder(tf.float32, [constants.BATCH_SIZE])

    # create variable scope for the network
    with tf.variable_scope('cnn') as scope:
        output = conv_nn(features)
        scope.reuse_variables()
        test_output = conv_nn(test_features)

    return output, test_output, test_features, test_labels


def get_loss(output, labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output))
    return loss


def compute_accuracy(output, labels):
    # function to compute accuracy 
    accuracy = tf.metrics.accuracy(labels, tf.argmax(output, axis=1))
    return accuracy


def get_optimizer(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=constants.LEARNING_RATE)
    train_op = optimizer.minimize(loss)

    return train_op
