from DataLoader import DataLoader
from helper import learning_rate_function
import Net
import tensorflow as tf
from math import ceil
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start_time = time.time()

batch_size = 64
max_epochs = 31
nr_classes = 4
data_set_size = 256
model_path = 'checkpoints'
log_path = 'logs'
save = True
print_float_operations = False

dataset_path = "C:/Users/Fabian/OneDrive - bwedu/Privat/MotionDetection_Dataset/SensorCollector_Time_Gravity/"

dataLoader = DataLoader(dataset_path, 4, batch_size, [0.8, 0.15])

steps_per_epoch = ceil(dataLoader.length(DataLoader.TRAIN) / batch_size)
val_steps = ceil(dataLoader.length(DataLoader.VAL) / batch_size)
test_steps = ceil(dataLoader.length(DataLoader.TEST) / batch_size)

# Declare input variables
x = tf.placeholder(tf.float32, [None, data_set_size, 4], name="x")
x_img = tf.reshape(x, [-1, 32, 32, 1])
y_ = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
learning_rate = tf.placeholder(tf.float64, name="learning_rate")
tf.summary.scalar('learning_rate', learning_rate)

y = Net.create_net(x_img, keep_prob, nr_classes, "testnet")

# softmax for output
softmax = tf.nn.softmax(y, name="softmax")

# Declare losses
cross_entropy = tf.losses.sparse_softmax_cross_entropy(y_, y)
tf.summary.scalar('cross_entropy', cross_entropy)

# Declare metrics
correct_prediction = tf.equal(tf.argmax(y, 1, output_type=tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# setup trainer
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(tf.losses.get_total_loss(True), global_step=global_step)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if save:
        checkpoint = tf.train.get_checkpoint_state(model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    if print_float_operations:
        # measure floating point operation for the model
        g = sess.graph
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        print("Flops: " + str(flops.total_float_ops))
        # **********************************************

    if save:
        # save graph def
        tf.train.write_graph(sess.graph.as_graph_def(), Net.graph_def_path, 'model.pbtxt', as_text=True)
        writer = tf.summary.FileWriter(log_path, sess.graph)

    iteration_cnt = 0

    for epoch in range(max_epochs):
        start_epoch = time.time()
        lr = learning_rate_function.learning_rate(epoch)
        for i in range(steps_per_epoch):
            batch_x, batch_y = dataLoader.next_batch(DataLoader.TRAIN)
            _, train_loss = sess.run([train_step, cross_entropy],
                                     feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5, learning_rate: lr})
            print("Step " + str(i) + "/" + str(steps_per_epoch) + ": " + str(train_loss), end='\r')
            if i % 50 == 0:
                merged = tf.summary.merge_all()
                summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5, learning_rate: lr})
                if save:
                    writer.add_summary(summary, iteration_cnt * batch_size)
            iteration_cnt += 1

        accuracy_agg = 0
        for i in range(val_steps):
            batch_x, batch_y = dataLoader.next_batch(DataLoader.VAL)
            accuracy_agg += accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        print('Epoch ' + str(epoch) + ', validation accuracy ' + str(
            round(1.0 * accuracy_agg / val_steps, 4)) + ' Time: ' +
              str(round(time.time() - start_epoch, 4)))
    if save:
        saver.save(sess, os.path.join(model_path, "testnet"), global_step=epoch)

    accuracy_agg = 0.0
    for i in range(test_steps):
        batch_x, batch_y = dataLoader.next_batch(DataLoader.TEST)
        accuracy_agg += accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    print('Final test accuracy %g' % (1.0 * accuracy_agg / test_steps))
    merged = tf.summary.merge_all()
    summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, learning_rate: lr})
    if save:
        writer.add_summary(summary, iteration_cnt * batch_size)
