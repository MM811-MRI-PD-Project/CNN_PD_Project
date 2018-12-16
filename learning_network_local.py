width = 128
height = 128
depth = 20
nLabel = 2

# Start TensorFlow InteractiveSession
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import input_3Dimage_local as input_dataset
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sess = tf.InteractiveSession()
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
#config.gpu_options.allow_growth = True
tf.Session(config=config)
#sess = tf.Session(config = config)


# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width*height*depth], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, nLabel],name='y_true')
y_true_cls = tf.argmax(y_true, axis = None, dimension=1)
is_training = tf.placeholder(tf.bool, [], name='is_training')
keep_prob = tf.placeholder(tf.float32)

# batch size
batch_size = 23
test_batch_size = 25
img_size_flat = width*height*depth

# Weight Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



# Convolution layer: output size = input size
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
# Pooling layer for the first 3 repeats:
def max_pool_4x4x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 2, 1], strides=[1, 4, 4, 2, 1], padding='SAME')
# Pooling layer for the last repeat:
def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')


## 1st Repeat:
# 1st repeat increase features: 1-->16
W_conv1 = weight_variable([3, 3, 3, 1, 16])
b_conv1 = bias_variable([16])
# Reshape the original input images to 4D:
x_image = tf.reshape(x, [-1,width,height,depth,1])
print(x_image.get_shape)
# First conv&relu layer:
conv1_res = conv3d(x_image, W_conv1)
h_conv1 = tf.nn.relu(conv1_res + b_conv1)
print(h_conv1.get_shape)
# Drop-out:
#h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)
#print(h_conv1_drop)
# First batch normalization layer:
h_bn1 = tf.contrib.layers.batch_norm(h_conv1, is_training= is_training, updates_collections= None)
print(h_bn1.get_shape)
# First pooling layer:
h_pool1 = max_pool_4x4x2(h_bn1)
print(h_pool1.get_shape)
# Output size shoule be --> 32*32*10


## 2nd Repeat:
# 2nd repeat increase features: 16-->32
W_conv2 = weight_variable([3, 3, 3, 16, 32])
b_conv2 = bias_variable([32])
# Second conv&relu layer:
conv2_res = conv3d(h_pool1, W_conv2)
h_conv2 = tf.nn.relu(conv2_res + b_conv2)
print(h_conv2.get_shape)
# Drop-out:
#h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)
#print(h_conv2_drop)
# Second batch normalization layer:
h_bn2 = tf.contrib.layers.batch_norm(h_conv2, is_training= is_training, updates_collections= None)
print(h_bn2.get_shape)
# Second pooling layer:
h_pool2 = max_pool_4x4x2(h_bn2)
print(h_pool2.get_shape)
# Output size shoule be --> 8*8*5


## 3rd Repeat:
# 3rd repeat increase features: 32-->64
W_conv3 = weight_variable([3, 3, 3, 32, 64])
b_conv3 = bias_variable([64])
# Third conv&relu layer:
conv3_res = conv3d(h_pool2, W_conv3)
h_conv3 = tf.nn.relu(conv3_res + b_conv3)
print(h_conv3.get_shape)
# Drop-out:
#h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)
#print(h_conv3_drop.get_shape)
# Third batch normalization layer:
h_bn3 = tf.contrib.layers.batch_norm(h_conv3, is_training= is_training, updates_collections= None)
print(h_bn3.get_shape)
# Third pooling layer:
h_pool3 = max_pool_4x4x2(h_bn3)
print(h_pool3.get_shape)
# Output size shoule be --> 2*2*3

## 4th Repeat:
# 2nd repeat increase features: 64-->128
W_conv4 = weight_variable([3, 3, 3, 64, 128])
b_conv4 = bias_variable([128])
# 4th conv&relu layer:
conv4_res = conv3d(h_pool3, W_conv4)
h_conv4 = tf.nn.relu(conv4_res + b_conv4)
print(h_conv4.get_shape)
# Drop-out:
#h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)
#print(h_conv4_drop)
# 4th batch normalization layer:
h_bn4 = tf.contrib.layers.batch_norm(h_conv4, is_training= is_training, updates_collections= None)
print(h_bn4.get_shape)
# 4th pooling layer:
h_pool4 = max_pool_2x2x2(h_bn4)
print(h_pool4.get_shape)
# Output size shoule be --> 1*1*1


## Densely Connected Layer
W_fc1 = weight_variable([1*1*1*128, 100])
b_fc1 = bias_variable([100])
h_pool4_flat = tf.reshape(h_pool4, [-1, 1*1*1*128])
print(h_pool4_flat.get_shape)

# 1st Fully-connected layer:
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
print(h_fc1.get_shape)

# Dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.get_shape)

# Readout Layer
W_fc2 = weight_variable([100, nLabel])
b_fc2 = bias_variable([nLabel])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print('y_conv shape:',y_conv.get_shape)
print('y_conv',y_conv)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_true,1))
#print('correct_prediction shape:',correct_prediction.get_shape)
#print('tf.argmax(y_conv)shape:',tf.argmax(y_conv,1).get_shape)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start the session:
sess.run(tf.global_variables_initializer())
accuracy_level ={"train_acc":[],"val_acc":[],"train_loss":[],"val_loss":[]}
csvfile = open('Result.csv','w',newline='')
result = csv.writer(csvfile,delimiter=',')
# Training:
data = input_dataset.get_train_data_MRI(sess,batch_size,validation_size=0.16)
testset = input_dataset.read_test_set(sess,test_batch_size)
#test_images = testset[0]
#test_labels = testset[1]
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(testset.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))
for i in range(420):
    x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
    x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
    #print("x_valid_batch:",x_valid_batch.shape)
    #print("y_valid_batch:",y_valid_batch.shape)
    x_batch = x_batch.reshape(batch_size, img_size_flat)
    x_valid_batch = x_valid_batch.reshape(batch_size, img_size_flat)

    feed_dict_train = {x: x_batch,y_true: y_true_batch,is_training: True,keep_prob: 0.7}
    feed_dict_validate = {x: x_valid_batch,y_true: y_valid_batch, is_training: True,keep_prob: 1.0}

    train_step.run(feed_dict=feed_dict_train)
    #if i % 2 ==0:
    val_loss = sess.run(cross_entropy, feed_dict= feed_dict_validate)
    train_loss = sess.run(cross_entropy, feed_dict= feed_dict_train)
    train_accuracy = sess.run(accuracy, feed_dict=feed_dict_train)
    val_accuracy = sess.run(accuracy, feed_dict=feed_dict_validate)
    epoch = i
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Train Loss: {3:.3f},Validation Loss: {4:.3f}"
    print(msg.format(epoch, train_accuracy,val_accuracy, train_loss, val_loss))
    accuracy_level["train_acc"].append(train_accuracy)
    accuracy_level["train_loss"].append(train_loss)
    accuracy_level["val_acc"].append(val_accuracy)
    accuracy_level["val_loss"].append(val_loss)
    result.writerow([epoch, train_accuracy,val_accuracy, train_loss, val_loss])
csvfile.close()
print("finish training, start testing:")


# Testing:
x_test_batch, y_test_batch, _,test_cls_batch = testset.test.next_batch(test_batch_size)
x_test_batch  = x_test_batch.reshape(test_batch_size, img_size_flat)
feed_dict_test = {x: x_test_batch, y_true: y_test_batch, is_training: True, keep_prob: 1.0}
test_accuracy = sess.run(accuracy, feed_dict=feed_dict_test)
print("test accuracy %g"%test_accuracy)

#def write_predictions(predict_logis):
#    result = pd.DataFrame(predict_logis, columns=classes)
#    result.loc[:, 'id'] = pd.Series(ids, index=result.index)
#    pred_file = 'predictions.csv'
#    result.to_csv(pred_file, index=False)


#result_test.writerow([test_images, test_ids,predict_logis, test_accuracy])
#write_predictions(test_images, test_ids)

#y = accuracy_level
#x = np.arange(6)
#plt.plot(x,y)
#plt.xlabel('iterations', fontsize=18)
#plt.ylabel('accuracy', fontsize=18)
#plt.show()
