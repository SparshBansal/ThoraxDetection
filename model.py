import tensorflow as tf

class Model:
    
    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]

    def get_output(self):
        return self.outputs[-1]

    def get_num_layers(self):
        return len(self.outputs) ;

    def get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        return '%sL%03d'%(self.name , layer+1)

    def get_num_inputs(self):
        return self.get_output().get_shape()[-1];

    def add_conv2d(self , kernel_size=5, output_channels=32 , stride=4 , stddev=1):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4 dimensional (batch, width, height, channels)"

        with tf.variable_scope(self.get_layer_str()):
            input_channels = self.get_num_inputs()
            
            # make variables for weights and biases 
            weight = tf.get_variable('weight' , shape=[kernel_size,kernel_size,input_channels,output_channels],
                    initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', shape=[output_channels], initializer=tf.constant(0.0))


            out = tf.nn.conv2d(self.get_output(), weight, stride=[1,stride,stride,1],padding='SAME')
            out = tf.nn.bias_add(out,bias);

        self.outputs.append(out)

    def add_relu(self):
        with tf.variable_scope(tf.get_layer_str()):
            out = tf.nn.relu(self.get_output())
        self.append(out)
    
    # add fully connected layer
    def add_fc(self, output_units):

        inputs = self.get_output();

        # flatten if needed
        if len(self.get_output().get_shape()) > 2: 
            # find input units
            input_shape = self.get_output().get_shape().as_list()
            inunits=1
            for dim in input_shape[1:]:
                inunits = inunits*dim

            inputs = tf.reshape(inputs, [-1,inunits])

        with tf.variable_scope(self.get_layer_str()):
            out = tf.layers.dense(inputs=inputs, units=output_units , activation=tf.nn.relu)

        self.outputs.append(out)

    
def conv_nn(features):
    
    model = Model('cnn' , features)
    
    # add layers accordingly
    model.add_fc(20)
    model.add_fc(10)

    return model.get_output()

def create_model(features,labels):

    # create variable scope for the network
    with tf.variable_scope('cnn') as scope:
        output = conv_nn(features)
        scope.reuse_variables()
    return output

def get_loss(output, labels):
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    return loss

def get_optimizer(loss):
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return train_op
