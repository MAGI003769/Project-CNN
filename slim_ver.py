# Try to construct the CNN #
# ------------------------ #
# In general, xdata represents images while ydata represents labels. #

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import tensorflow.contrib.slim as slim
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Load data
train_dir = '.\\images\\doge.jpg'
test_dir = ''
train_x = np.asarray([cv2.imread(train_dir)])
print(np.asarray(cv2.imread(train_dir)))

# Set model parameters
batch_size = 10
learning_rate = 0.005
test_size = 20
im_width = train_x[0].shape[0]
im_height = train_x[0].shape[1]
labels_size = 10
num_channels = 3
train_epochs = 500
keep_prob = 1

conv1_output = 100
conv2_output = 200
conv3_output = 300
conv4_output = 400
conv5_output = 500
conv6_output = 600
conv7_output = 700
conv_size = 2

max_pool_size1 = 2
max_pool_size2 = 2
max_pool_size3 = 2
max_pool_size4 = 2
max_pool_size5 = 2
max_pool_size6 = 2

fully_connected1_size = 100

# Declare model placeholders
x_input_shape = (None, im_width, im_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_label = tf.placeholder(tf.int32, shape=(None))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape = shape, dtype=tf.float32)
    return tf.Variable(initial)


# Initialize Model Operations
def my_CNN(input):
    # 1st layer: 100C3-MP2
    conv_1 = slim.conv2d(input, 100, [3, 3], 1, padding='SAME', scope='conv1',activation_fn=tf.nn.relu)
    max_pool1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')

    # 2nd layer: 200C2-MP2
    conv_2 = slim.conv2d(max_pool1, 200, [2, 2], 1, padding='SAME', scope='conv2',activation_fn=tf.nn.relu)
    max_pool2 = max_pool_1 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')

    # 3rd layer: 300C2-MP2
    conv_3 = slim.conv2d(max_pool2, 300, [2, 2], 1, padding='SAME', scope='conv3',activation_fn=tf.nn.relu)
    max_pool3 = max_pool_1 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')

    # 4th layer: 400C2-MP2
    conv_4 = slim.conv2d(max_pool3, 400, [2, 2], 1, padding='SAME', scope='conv4',activation_fn=tf.nn.relu)
    max_pool4 = max_pool_1 = slim.max_pool2d(conv_4, [2, 2], [2, 2], padding='SAME')
    
    # 5th layer: 500C2-MP2
    conv_5 = slim.conv2d(max_pool4, 500, [2, 2], 1, padding='SAME', scope='conv5',activation_fn=tf.nn.relu)
    max_pool5 = max_pool_1 = slim.max_pool2d(conv_5, [2, 2], [2, 2], padding='SAME')

    # 6th layer: 600C2-MP2
    conv_6 = slim.conv2d(max_pool5, 600, [2, 2], 1, padding='SAME', scope='conv6',activation_fn=tf.nn.relu)
    max_pool6 = max_pool_1 = slim.max_pool2d(conv_6, [2, 2], [2, 2], padding='SAME')

    # 7th layer: 700C2
    conv_7 = slim.conv2d(max_pool6, 700, [2, 2], 1, padding='SAME', scope='conv7',activation_fn=tf.nn.relu)

    # Flat the output from conv layers for next fully connected layers
    flatten = slim.flatten(conv_7)
    
    # 1st fully connected layer
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                                   activation_fn=tf.nn.tanh, scope='fc1')

    # 2nd fully connected layer
    model_output = slim.fully_connected(slim.dropout(fc1, keep_prob), labels_size,
                                   activation_fn=None, scope='fc2')

    return(model_output)

model_output = my_CNN(x_input)

# Declare Loss function (softmax cross entropy)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_label))

# Creat a prediction function
prediction = tf.nn.softmax(model_output)

# Create an optimizer
my_optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = my_optimizer.minimize(loss)

# Calculate accuracy function
# In this function, batch_prediction is the ouput result from the CNN
# while labels are the real label stored in dataset which trains the model
def get_acc(logits, labels):
    batch_predictions = np.argmax(logits, axis=1)
    bingo = np.sum(np.equal(batch_predictions, labels))
    return(100. * bingo/batch_predictions.shape[0])

# Run the model
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    summary, predicted = sess.run([model_output, prediction], {x_input: train_x})
    for i in range(10):
        print(summary[0][i], predicted[0][i])

writer.close()