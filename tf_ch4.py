# -*- coding: utf-8 -*-
# v1 호환성 맞추기
import tensorflow.compat.v1 as tf
import numpy as np

# MNIST 데이터를 변수에 설정
mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape([-1, 784])
test_images  = test_images.reshape([-1,784])
train_images = train_images / 255.
test_images  = test_images / 255.
print(train_images[0])

print('train_images.shape : ', train_images.shape)

tf.disable_v2_behavior()

nb_classes = 10

# 변수들을 설정한다.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross-entropy 모델을 설정한다.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# 학습된 모델이 얼마나 정확한지를 출력한다.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

training_epochs = 50
batch_size = 100

import matplotlib.pyplot as plt
import random

# 경사하강법으로 모델을 학습한다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(train_images.shape[0] / batch_size)

        for i in range(total_batch):
            s_idx = int(train_images.shape[0] * i / total_batch)
            e_idx = int(train_images.shape[0] * (i+1) / total_batch)

            batch_xs = train_images[s_idx : e_idx]
            batch_ys = train_labels[s_idx : e_idx]

            Y_one_hot = np.eye(nb_classes)[batch_ys]
            _,c = sess.run([train_step, cross_entropy], feed_dict={x:batch_xs, y_:Y_one_hot})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    Y_one_hot = np.eye(nb_classes)[test_labels]
    print("Accuracy : ", accuracy.eval(session=sess, feed_dict={x:test_images, y_:Y_one_hot}))

    r = random.randint(0, test_images.shape[0] - 1)
    print('label : ', test_labels[r:r + 1])
    print('Prediction : ', sess.run(tf.argmax(y, 1), feed_dict={x: test_images[r:r + 1]}))
    plt.imshow(test_images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()


