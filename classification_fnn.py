from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from competition_utility import dafaset_loader as dl

import tensorflow as tf
import numpy as np

FLAGS = None


def create_model():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 64*64*3])
    initial_w = tf.truncated_normal([64*64*3, 2000], stddev=0.1, dtype=tf.float32)
    w = tf.Variable(initial_w)
    initial_b = tf.truncated_normal([2000], stddev=0.1, dtype=tf.float32)
    b = tf.Variable(initial_b)
    h = tf.nn.sigmoid(tf.matmul(x, w) + b)

    initial_w2 = tf.truncated_normal([2000, 2], stddev=0.1, dtype=tf.float32)
    w2 = tf.Variable(initial_w2)
    initial_b2 = tf.truncated_normal([2], stddev=0.1, dtype=tf.float32)
    b2 = tf.Variable(initial_b2)
    y = tf.nn.softmax(tf.matmul(h, w2) + b2)

    return x, y


def main(_):
    dirs = ["target", "others2"]
    labels = [dl.DataSet.TARGET, dl.DataSet.OTHERS]
    loader = dl.DatasetLoader(dirs, labels)
    train, test = loader.load_train_test()
    train.print_information()
    test.print_information()

    gpu = False

    x, y = create_model()

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    gpu_config = tf.ConfigProto(
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
      device_count={'GPU': 0})
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(100):
        for image in train(batch_size=32):
            batch_xs, batch_ys = image.images, image.labels
            batch_xs = batch_xs.reshape((len(batch_xs), 64*64*3))

            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if _ % 10 == 0:
            print("Train", sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
            print("Test", sess.run(cross_entropy, feed_dict={x: test.images.reshape((test.images.shape[0], 64*64*3)), y_: test.labels}))

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test.images.reshape((len(test.images), 64*64*3)),
                                      y_: test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
