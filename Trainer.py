from DataLoader import DataLoader
import tensorflow.contrib.slim as slim
import tensorflow as tf
from math import ceil
import time

start_time = time.time()


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


batch_size = 64
max_epochs = 20
nr_classes = 4
data_set_size = 64
model_path = 'checkpoints'
log_path = 'logs'

dataLoader = DataLoader("data/", 4, batch_size, [0.8, 0.15])

steps_per_epoch = ceil(dataLoader.length(DataLoader.TRAIN) / batch_size)
val_steps = ceil(dataLoader.length(DataLoader.VAL) / batch_size)
test_steps = ceil(dataLoader.length(DataLoader.TEST) / batch_size)

# Declare input variables
x = tf.placeholder(tf.float32, [None, data_set_size, 4])
x_img = tf.reshape(x, [-1, 16, 16, 1])
y_ = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)

y = create_net(x_img, keep_prob, nr_classes, "testnet")

# Declare losses
cross_entropy = tf.losses.sparse_softmax_cross_entropy(y_, y)
tf.summary.scalar('cross_entropy', cross_entropy)

# Declare metrics
correct_prediction = tf.equal(tf.argmax(y, 1, output_type=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# setup trainer
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(0.0001).minimize(tf.losses.get_total_loss(True), global_step=global_step)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    writer = tf.summary.FileWriter(log_path, sess.graph)
    iteration_cnt = 0

    print("Start time: " + str(time.time() - start_time))

    for epoch in range(max_epochs):
        start_epoch = time.time()
        for i in range(steps_per_epoch):
            batch_x, batch_y = dataLoader.next_batch(DataLoader.TRAIN)
            _, train_loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            print("Step " + str(i) + "/" + str(steps_per_epoch) + ": " + str(train_loss), end='\r')
            if i % 50 == 0:
                merged = tf.summary.merge_all()
                summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
                writer.add_summary(summary, iteration_cnt * batch_size)
            iteration_cnt += 1

        accuracy_agg = 0
        for i in range(val_steps):
            batch_x, batch_y = dataLoader.next_batch(DataLoader.VAL)
            accuracy_agg += accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        print('Epoch ' + str(epoch) + ', validation accuracy ' + str(
            round(1.0 * accuracy_agg / val_steps, 4)) + ' Time: ' +
              str(round(time.time() - start_epoch, 4)))
    # saver.save(sess, os.path.join(model_path, "testnet"), global_step=epoch)

    accuracy_agg = 0.0
    for i in range(test_steps):
        batch_x, batch_y = dataLoader.next_batch(DataLoader.TEST)
        accuracy_agg += accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    print('Final test accuracy %g' % (1.0 * accuracy_agg / test_steps))
    merged = tf.summary.merge_all()
    summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    writer.add_summary(summary, iteration_cnt * batch_size)
