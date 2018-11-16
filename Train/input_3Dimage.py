# A script to load images and make batch.
# Dependency: 'nibabel' to load MRI (NIFTI) images
# Reference: http://blog.naver.com/kjpark79/220783765651

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
tf.app.flags.DEFINE_integer('depth', 121,
                            """depth of image""")

# user selection
tf.app.flags.DEFINE_string('data_dir', '/Users/DXX/Desktop/UACLASS/MM811/project/Datas/NII_TEST_TRAIN',
                           """Directory""")
tf.app.flags.DEFINE_integer('num_class', 2,
                            """classes of image""")
FLAGS = tf.app.flags.FLAGS

batch_index = 0
filenames = []

def get_filenames(data_set):
    global filenames
    labels = ['wmHC','wmPD']

    for i, label in enumerate(labels):
      
        filelist = os.listdir(FLAGS.data_dir  + '/' + data_set + '/' + label)
        for filename in filelist:
            if filename.endswith('.nii'):
                filenames.append([label + '/' + filename, i])

    random.shuffle(filenames)


def get_data_MRI(sess, data_set, batch_size):
    global batch_index, filenames

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
