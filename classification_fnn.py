"""
    This is a program for training of new laboratory members.
    新人研修用の画像クラス分類プログラムです．

    Author: Kazuki Takaishi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from competition_utility import dafaset_loader as dl

import tensorflow as tf
import numpy as np


def create_model():
    # Create the free connected layers
    # 全結合のモデルを生成します
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


def load_dataset():
    dirs = ["images/kobayashi", "images/others"]
    labels = [dl.DataSet.TARGET, dl.DataSet.OTHERS]
    loader = dl.DatasetLoader(dirs, labels)
    return loader.load_train_test()


def main(_):

    # Load training and test data
    # 訓練とテストデータを読み込みます
    train, test = load_dataset()

    # Whether or not using a GPU
    # GPUを使用するか
    gpu = False

    # Create a model
    # モデルの生成
    # 入力用のプレースホルダと最終層が返却される
    x, y = create_model()

    # Define output place holder
    # 出力のプレースホルダを定義します．ここに教師データが入ります．
    y_ = tf.placeholder(tf.float32, [None, 2])

    # Set loss function and optimizer
    # 誤差関数とオプティマイザの設定をします
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 精度の算出をします
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize session
    # セッションの初期化をします
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9), device_count={'GPU': 0})
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train model
    # モデルの訓練
    epochs = 100
    batch_size = 32
    input_size = 64 * 64 * 3
    for epoch in range(epochs):
        for image in train(batch_size=batch_size):
            batch_images, batch_labels = image.images, image.labels
            batch_images = batch_images.reshape((batch_images.shape[0], input_size))

            sess.run(train_step, feed_dict={x: batch_images, y_: batch_labels})
        if epoch % 10 == 0:
            loss_train = sess.run(cross_entropy, feed_dict={x: train.images.reshape((train.images.shape[0], input_size)),
                                                           y_: train.labels})
            loss_test = sess.run(cross_entropy, feed_dict={x: test.images.reshape((test.images.shape[0], input_size)),
                                                           y_: test.labels})
            accuracy_train = sess.run(accuracy, feed_dict={x: train.images.reshape((train.images.shape[0], input_size)),
                                                           y_: train.labels})
            accuracy_test = sess.run(accuracy, feed_dict={x: test.images.reshape((len(test.images), input_size)),
                                                           y_: test.labels})
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)

    # Test trained model
    # 訓練済みモデルの評価
    loss_test = sess.run(cross_entropy, feed_dict={x: test.images.reshape((test.images.shape[0], input_size)),
                                                   y_: test.labels})
    accuracy_test = sess.run(accuracy, feed_dict={x: test.images.reshape((len(test.images), input_size)),
                                                   y_: test.labels})
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
