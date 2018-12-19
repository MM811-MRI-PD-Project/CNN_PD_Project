import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import nibabel as nib
from sklearn.utils import shuffle
import glob

tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_integer('width', 128,
                            """width of image""")
tf.app.flags.DEFINE_integer('height', 128,
                            """height of image""")
tf.app.flags.DEFINE_integer('depth', 20,
                            """depth of image""")


FLAGS = tf.app.flags.FLAGS
batch_index = 0
image_size = 128

# Please change the train & test path:

train_path = "/home/luchen/mm811/DATASET/GM/train/"
#train_path = "/Users/luchenliu/Documents/UAlberta_CS_MM/deep_learning/project/mm811/gm_labeldatas/train/"
#test_path = "/Users/luchenliu/Documents/UAlberta_CS_MM/deep_learning/project/mm811/gm_labeldatas/test/"
test_path = "/home/luchen/mm811/DATASET/GM/test/"
classes = ['gmHC', 'gmPD']
num_classes = len(classes)

def load_train(sess,batch_size):
    images = np.array([], np.float32)
    labels = []
    ids = []
    cls = []
    print('Reading training images')
    for fld in classes:
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*.nii')
        files = glob.glob(path)
        for fl in files:
            FA_org= nib.load(fl)
            FA_data = FA_org.get_data()
            FA_data = FA_data[:,:,50:70]

            resized_image = tf.image.resize_images(images=FA_data, size=(FLAGS.width,FLAGS.height), method=1)
            image = sess.run(resized_image)
            images = np.append(images, np.asarray(image, dtype='float32'))
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase_train = os.path.basename(fl)
            ids.append(flbase_train)
            cls.append(fld)
    print(images.shape)
    images = images.reshape(len(labels), FLAGS.height * FLAGS.width * FLAGS.depth)
    print(images.shape)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    return images, labels, ids, cls


def load_test(sess,test_batch_size):
    test_images = np.array([], np.float32)
    test_labels = []
    test_ids = []
    test_cls = []
    print('Reading test images')
    for fld in classes:   
        test_index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, test_index))
        joined_test_path = os.path.join(test_path, fld, '*.nii')
        test_files = glob.glob(joined_test_path)
        print(len(test_files))
        for test_fl in test_files:
            test_FA_org= nib.load(test_fl)
            test_FA_data = test_FA_org.get_data()
            test_FA_data = test_FA_data[:,:,50:70]

            test_resized_image = tf.image.resize_images(images=test_FA_data, size=(FLAGS.width,FLAGS.height), method=1)
            test_image = sess.run(test_resized_image)
            test_images = np.append(test_images, np.asarray(test_image, dtype='float32'))
            test_label = np.zeros(len(classes))
            test_label[test_index] = 1.0
            test_labels.append(test_label)
            flbase_test = os.path.basename(test_fl)
            test_ids.append(flbase_test)
            test_cls.append(fld)
    print(test_images.shape)
    test_images = test_images.reshape(len(test_labels), FLAGS.height * FLAGS.width * FLAGS.depth)
    test_labels = np.array(test_labels)
    test_ids = np.array(test_ids)
    test_cls = np.array(test_cls)
    return test_images, test_labels, test_ids, test_cls


#def load_test (sess):
#  path = os.path.join(test_path, '*.nii')
#  files = sorted(glob.glob(path))
#  X_test = np.array([], np.float32)
#  X_test_id = []
#  print("Reading test images")
#  for fl in files:
#     flbase = os.path.basename(fl)
#     FA_org= nib.load(fl)
#     FA_data = FA_org.get_data()
#     FA_data = FA_data[:,:,50:70]
#
#     resized_img = tf.image.resize_images(images=FA_data, size=(FLAGS.width,FLAGS.height), method=1)
#     img = sess.run(resized_img)
#     X_test = np.append(X_test, np.asarray(img, dtype='float32'))
#     X_test_id.append(flbase)
#
#  X_test = X_test.reshape(len(X_test_id), FLAGS.height * FLAGS.width * FLAGS.depth)
#  X_test = X_test / 255
#  return X_test, X_test_id

class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def ids(self):
        return self._ids
    @property
    def cls(self):
        return self._cls
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed


#class TestDataSet(object):
#    def __init__(self, test_images, test_labels, test_ids, test_cls):
#        self._num_test_examples = images.shape[0]
#        test_images = test_images.astype(np.float32)
#        test_images = np.multiply(test_images, 1.0 / 255.0)
#        self._test_images = test_images
#        self._test_labels = test_images
#        self._test_ids = test_ids
#        self._test_cls = test_cls
#        #self._epochs_completed = 0
#        self._index_in_test_epoch = 0
#    @property
#    def test_images(self):
#        return self._test_images
#    @property
#    def test_labels(self):
#        return self._test_labels
#    @property
#    def test_ids(self):
#        return self._test_ids
#    @property
#    def test_cls(self):
#        return self._test_cls
#    @property
#    def num_test_examples(self):
#        return self._num_test_examples
#    @property
#    def epochs_completed(self):
#        return self._epochs_completed


def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
        self._epochs_completed += 1
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def get_train_data_MRI(sess, batch_size, validation_size=0):
    class DataSet(object):
        pass
    data_sets = DataSet()
    images, labels, ids, cls = load_train(sess,batch_size)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)
    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_ids = ids[:validation_size]
    validation_cls = cls[:validation_size]
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]
    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)
    return data_sets


#def read_test_set(sess):
#  test_images, test_labels, test_ids, test_cls  = load_test(sess)
#  return test_images, test_labels

def read_test_set(sess,test_batch_size):
    class DataSet(object):
        pass
    test_data_sets = DataSet()
    #test_images, test_labels, test_ids, test_cls  = load_test(sess)
    test_images,test_labels, test_ids, test_cls = load_test(sess, test_batch_size)
    test_images,test_labels, test_ids, test_cls = shuffle(test_images,test_labels, test_ids, test_cls)
    print("len of labels in testset:",len(test_labels))
    test_data_sets = DataSet(test_images,test_labels, test_ids, test_cls)
    print("len of labels in testset:",len(test_labels))
    return test_data_sets

