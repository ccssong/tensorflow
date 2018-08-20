import tensorflow as tf
import numpy as np
import glob
from sklearn.model_selection import KFold

n_dim = 39
n_classes = 2
n_hid1 = 300
n_hid2 = 200
n_hid3 = 100

training_epochs = 200
learning_rate = 0.01
sd = 1 / np.sqrt(n_dim)

X = tf.placeholder(tf.float32, [None, 48, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hid1], mean=0, stddev=sd), name="w1")
b_1 = tf.Variable(tf.random_normal([n_hid1], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.relu(tf.matmul(X,W_1)+b_1)

W_2 = tf.Variable(tf.random_normal([n_hid2, n_hid3], mean = 0, stddev=sd),name="w3")
b_2 = tf.Variable(tf.random_normal([n_hid2], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2)+b_2)

W_3 = tf.Variable(tf.random_normal([n_hid2, n_hid3], mean = 0, stddev=sd), name="w3")
b_3 = tf.Variable(tf.random_normal([n_hid3], mean=0, stddev=sd),name="b3")
h_3 = tf.nn.relu(tf.matmul(h_2, W_3)+b_3)

keep_prob = tf.placeholder(tf.float32)
h_3_drop = tf.nn.dropout(h_3, keep_prob)

W = tf.Variable(tf.randomnormal([n_hid3, n_classes], mean=0, stddev=sd), name="w")
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
y_ = tf.nn.softmax(tf.matmul(h_3_drop, W) + b)

cost = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(Y*tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

is_correct = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mkean(tf.cast(is_correct, tf.float32))

#k-cross validation해준다



sess = tf.Session()
init = tf.global_variable_initializer()
sess.run(init)



#for epoch in range(1000)