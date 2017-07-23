# Try to construct the CNN #
# ------------------------ #
# In general, xdata represents images while ydata represents labels. #

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Load data
train_dir = '.\\images\\doge.jpg'
test_dir = ''
train_x = np.asarray([cv2.imread(train_dir)])

# Set model parameters
batch_size = 1
learning_rate = 0.005
test_size = 20
im_width = train_x[0].shape[0]
im_height = train_x[0].shape[1]
print(im_height)
labels_size = 10
num_channels = 3
train_epochs = 500

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
x_input_shape = (batch_size, im_width, im_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_label = tf.placeholder(tf.int32, shape=(batch_size))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape = shape, dtype=tf.float32)
    return tf.Variable(initial)

# Convolution variables
conv1_weight = weight_variable([3, 3, num_channels, conv1_output])
conv1_bias = bias_variable([conv1_output])

conv2_weight = weight_variable([conv_size, conv_size, conv1_output, conv2_output])
conv2_bias = bias_variable([conv2_output])

conv3_weight = weight_variable([conv_size, conv_size, conv2_output, conv3_output])
conv3_bias = bias_variable([conv3_output])

conv4_weight = weight_variable([conv_size, conv_size, conv3_output, conv4_output])
conv4_bias = bias_variable([conv4_output])

conv5_weight = weight_variable([conv_size, conv_size, conv4_output, conv5_output])
conv5_bias = bias_variable([conv5_output])

conv6_weight = weight_variable([conv_size, conv_size, conv5_output, conv6_output])
conv6_bias = bias_variable([conv6_output])

conv7_weight = weight_variable([conv_size, conv_size, conv6_output, conv7_output])
conv7_bias = bias_variable([conv7_output])

# Fully connected variables
resulting_width = im_width // (max_pool_size1 * max_pool_size2 * max_pool_size3 * max_pool_size4 * max_pool_size5 * max_pool_size6)
resulting_height = im_height // (max_pool_size1 * max_pool_size2 * max_pool_size3 * max_pool_size4 * max_pool_size5 * max_pool_size6)
full1_input_size = resulting_height * resulting_width * conv7_output
full1_weight = weight_variable([full1_input_size, fully_connected1_size])
full1_bias = weight_variable([fully_connected1_size])
full2_weight = weight_variable([fully_connected1_size, labels_size])
full2_bias = weight_variable([labels_size])


# Initialize Model Operations
def my_CNN(input):
	# 1st layer: 100C3-MP2
	conv1 = tf.nn.conv2d(input, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
	max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
		                        strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')
	print('max_pool1:', max_pool1.shape)

	# 2nd layer: 200C2-MP2
	conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
	max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
		                        strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')
	print('max_pool2:', max_pool2.shape)

	# 3rd layer: 300C2-MP2
	conv3 = tf.nn.conv2d(max_pool2, conv3_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_bias))
	max_pool3 = tf.nn.max_pool(relu3, ksize=[1, max_pool_size3, max_pool_size3, 1],
		                        strides=[1, max_pool_size3, max_pool_size3, 1], padding='SAME')
	print('max_pool3:', max_pool3.shape)

	# 4th layer: 400C2-MP2
	conv4 = tf.nn.conv2d(max_pool3, conv4_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_bias))
	max_pool4 = tf.nn.max_pool(relu4, ksize=[1, max_pool_size4, max_pool_size4, 1],
		                        strides=[1, max_pool_size4, max_pool_size4, 1], padding='SAME')
	print('max_pool4:', max_pool4.shape)

	# 5th layer: 500C2-MP2
	conv5 = tf.nn.conv2d(max_pool4, conv5_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_bias))
	max_pool5 = tf.nn.max_pool(relu5, ksize=[1, max_pool_size5, max_pool_size5, 1],
		                        strides=[1, max_pool_size5, max_pool_size5, 1], padding='SAME')
	print('max_pool5:', max_pool5.shape)

	# 6th layer: 600C2-MP2
	conv6 = tf.nn.conv2d(max_pool5, conv6_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_bias))
	max_pool6 = tf.nn.max_pool(relu6, ksize=[1, max_pool_size6, max_pool_size6, 1],
		                        strides=[1, max_pool_size6, max_pool_size6, 1], padding='SAME')
	print('max_pool6:', max_pool6.shape)

	# 7th layer: 700C2
	conv7 = tf.nn.conv2d(max_pool6, conv7_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_bias))
	print('relu7:', relu7.shape)

	# Flat the output from conv layers for next fully connected layers
	final_conv_shape = relu7.get_shape().as_list()
	flat_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
	flat_output = tf.reshape(relu7, [final_conv_shape[0], flat_shape])

	# 1st fully connected layer
	fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

	# 2nd fully connected layer
	model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

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
def get_acc(logists, labels):
	batch_predictions = np.argmax(logists, axis=1)
	bingo = np.sum(np.equal(batch_predictions, labels))
	return(100. * bingo/batch_predictions.shape[0])

# Run the model
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(prediction, {x_input: train_x}))