import tensorflow.contrib.slim as slim
import tensorflow as tf

nr_classes = 4
data_set_size = 64
model_path = 'checkpoints'


def create_net(input, keep_prob, nr_classes, name):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.conv2d(input, 16, [3, 3], padding='SAME', scope=name + "/convolutional_part/conv1")
        net = slim.max_pool2d(net, [2, 2], [2, 2], scope=name + "/convolutional_part/pool1")
        net = slim.conv2d(net, 32, [4, 4], padding='SAME', scope=name + "/convolutional_part/conv2")
        net = slim.max_pool2d(net, [4, 4], [2, 2], scope=name + "/convolutional_part/pool2")
        net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope=name + "/convolutional_part/conv3")
        net = slim.max_pool2d(net, [3, 3], [2, 2], scope=name + "/convolutional_part/pool3")

        net = slim.flatten(net, scope=name + "/flatten")
        net = slim.fully_connected(net, 1024, scope=name + "/fc_1")
        net = slim.dropout(net, keep_prob, scope=name + "/dropout_1")
        net = slim.fully_connected(net, 2048, scope=name + "/fc_2")
        net = slim.dropout(net, keep_prob, scope=name + "/dropout_2")
        return slim.fully_connected(net, nr_classes, scope=name + "/fc_out")
