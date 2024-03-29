import tensorflow.contrib.slim as slim
import tensorflow as tf

graph_def_path = "graph_def"
nr_classes = 4
data_set_size = 256
model_path = 'checkpoints'


def create_net(input, keep_prob, nr_classes, name):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.crelu,
                        weights_initializer=tf.uniform_unit_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        net = slim.conv2d(input, 32, [3, 3], padding='SAME', scope=name + "/convolutional_part/conv1")
        net = slim.max_pool2d(net, [2, 2], [2, 2], scope=name + "/convolutional_part/pool1")
        net = slim.conv2d(net, 32, [4, 4], padding='SAME', scope=name + "/convolutional_part/conv2")
        net = slim.max_pool2d(net, [4, 4], [2, 2], scope=name + "/convolutional_part/pool2")
        net = slim.conv2d(net, 32, [3, 3], padding='SAME', scope=name + "/convolutional_part/conv3")
        net = slim.max_pool2d(net, [3, 3], [2, 2], scope=name + "/convolutional_part/pool3")

        net = slim.flatten(net, scope=name + "/fully/flatten")
        net = slim.fully_connected(net, 2048, scope=name + "/fully/fc_1")
        net = slim.dropout(net, keep_prob, scope=name + "/fully/dropout_1")
        net = slim.fully_connected(net, 2048, scope=name + "/fully/fc_2")
        net = slim.dropout(net, keep_prob, scope=name + "/fully/dropout_2")
        return slim.fully_connected(net, nr_classes, activation_fn=None, scope=name + "/fully/fc_out")
