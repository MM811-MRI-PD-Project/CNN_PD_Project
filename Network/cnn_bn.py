import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import nibabel as nib

tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_integer('width', 128,
                            """width of image""")
tf.app.flags.DEFINE_integer('height', 128,
                            """height of image""")
tf.app.flags.DEFINE_integer('depth', 20,
                            """depth of image""")

# user selection
tf.app.flags.DEFINE_string('data_dir', '/content/drive/My Drive/APP/NII_Test_Train',
                           """Directory""")
tf.app.flags.DEFINE_integer('num_class', 2,
                            """classes of image""")
FLAGS = tf.app.flags.FLAGS

batch_index = 0
filenames = []
train_filenames = []
test_filenames = []

def get_filenames(data_set):
    global filenames
    labels = ['wmHC','wmPD']
    
    print(data_set, len(filenames))
    for i, label in enumerate(labels):
        filelist = os.listdir(FLAGS.data_dir  + '/' + data_set + '/' + label)
        for filename in filelist:
            if filename.endswith('.nii'):
                  filenames.append([label + '/' + filename, i])

    random.shuffle(filenames)


def get_data_MRI(sess, data_set, batch_size):
    global batch_index, filenames
    
    
    if data_set == 'test':
      filenames = test_filenames
    else:
      filenames = train_filenames
      
    if len(filenames) == 0: get_filenames(data_set) 
    max = len(filenames)
    #print(max)

    begin = batch_index
    end = batch_index + batch_size

    if end >= max:
        end = max
        batch_index = 0

    x_data = np.array([], np.float32)
    y_data = np.zeros((batch_size, FLAGS.num_class)) # zero-filled list for 'one hot encoding'
    index = 0

    for i in range(begin, end):
        
        imagePath = FLAGS.data_dir + '/' + data_set + '/' + filenames[i][0]
        FA_org = nib.load(imagePath)
        FA_data = FA_org.get_data()  # 256x256x40; numpy.ndarray
        FA_data = FA_data[...,0:-1:6]
        #print(FA_data.shape)
        # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
        resized_image = tf.image.resize_images(images=FA_data, size=(FLAGS.width,FLAGS.height), method=1)

        image = sess.run(resized_image)  # (256,256,40)
        x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
        #print(x_data.shape)
        y_data[index][filenames[i][1]] = 1  # assign 1 to corresponding column (one hot encoding)
        index += 1

    batch_index += batch_size  # update index for the next batch
    #print(batch_index)
    x_data_ = x_data.reshape(batch_size, FLAGS.height * FLAGS.width * FLAGS.depth)

    return x_data_, y_data

# A simple CNN to predict certain characteristics of the human subject from MRI images.
# 3d convolution is used in each layer.
# Reference: https://www.tensorflow.org/get_started/mnist/pros, http://blog.naver.com/kjpark79/220783765651
# Adjust needed for your dataset e.g., max pooling, convolution parameters, training_step, batch size, etc

width = 128
height = 128
depth = 20
nLabel = 2
'''
from google.colab import files
src = list(files.upload().values())[0]
open('input_3Dimage.py','wb').write(src)
import input_3Dimage
'''

# Start TensorFlow InteractiveSession
#from input_3Dimage import get_data_MRI
#import tensorflow as tf
sess = tf.InteractiveSession()

# Placeholders (MNIST image:28x28pixels=784, label=10)
x = tf.placeholder(tf.float32, shape=[None, width*height*depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]
is_training = tf.placeholder(tf.bool, [], name='is_training')

## Weight Initialization
# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')

## First Convolutional Layer
# Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
x_image = tf.reshape(x, [-1,width,height,depth,1]) # [-1,28,28,1]
print(x_image.get_shape) # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1

# x_image * weight tensor + bias -> apply ReLU -> apply max-pool
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
print(h_conv1.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32

h_bn1 = tf.contrib.layers.batch_norm(h_conv1, is_training= is_training, updates_collections= None)
print(h_bn1.get_shape)

h_pool1 = max_pool_2x2(h_bn1)  # apply max-pool 
print(h_pool1.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32




## Second Convolutional Layer
# Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
b_conv2 = bias_variable([64]) # [64]

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
print(h_conv2.get_shape) # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64

h_bn2 = tf.contrib.layers.batch_norm(h_conv2, is_training= is_training, updates_collections= None)
print(h_bn2.get_shape)

h_pool2 = max_pool_2x2(h_bn2)  # apply max-pool 
print(h_pool2.get_shape) # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64

## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image
W_fc1 = weight_variable([8*8*2*64, 1024])  # [7*7*64, 1024]
b_fc1 = bias_variable([1024]) # [1024]]

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*2*64])  # -> output image: [-1, 7*7*64] = 3136
print(h_pool2_flat.get_shape)  # (?, 2621440)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob )
print(h_fc1_drop.get_shape)  # -> output: 1024

## Readout Layer
W_fc2 = weight_variable([1024, nLabel]) # [1024, 10]
b_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  # -> output: 10

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
for i in range(50):
    batch = get_data_MRI(sess,'train',23)
    # Logging every 100th iteration in the training process.
    if i%5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], is_training: False,keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], is_training: True, keep_prob: 0.5})



# Evaulate our accuracy on the test data
testset = get_data_MRI(sess,'test',30)
print("test accuracy %g"%accuracy.eval(feed_dict={x: testset[0], y_: testset[1], is_training:False, keep_prob: 1.0}))
