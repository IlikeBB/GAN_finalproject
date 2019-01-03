import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
import numpy as np
import os, pdb
import cv2
import numpy as np
import random as rn
import tensorflow as tf
import threading
import time

global n_classes, ema_gp
ema_gp = []
n_classes = 40

def activation(x,name="activation"):
    return tf.nn.relu(x, name=name)
    
def conv2d(name, l_input, w, b, s, p):
    l_input = tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p, name=name)
    l_input = l_input+b

    return l_input

def max_pool(name, l_input, k, s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

def norm(l_input, lsize=4, name="lrn"):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def batchnorm(conv, out_channels):
    with tf.variable_scope('bn'):
      mean, var = tf.nn.moments(conv, axes=[0,1,2])
      beta = tf.Variable(tf.zeros([out_channels]), name="beta")
      gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1), name='gamma')

      batch_norm = tf.nn.batch_norm_with_global_normalization(
          conv, mean, var, beta, gamma, 0.001,
          scale_after_normalization=True)
      return batch_norm
  
def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def initializer(in_filters, out_filters, name):
    w1 = tf.get_variable(name+"W", [5, 5, in_filters, out_filters], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable(name+"B", [out_filters], initializer=tf.truncated_normal_initializer())
    return w1, b1
  
def residual_block(in_x, in_filters, out_filters, stride, isDownSampled, name):
    global ema_gp
    # first convolution layer
    if isDownSampled:
      in_x = tf.nn.avg_pool(in_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
      
      
    x = batchnorm(in_x, in_filters)
    x = activation(x)
    w1, b1 = initializer(in_filters, in_filters, name+"first_res")
    x = conv2d(name+'r1', x, w1, b1, 1, "SAME")
    
    
    
    # second convolution layer
    x = batchnorm(x, in_filters)
    x = activation(x)
    w2, b2 = initializer(in_filters, out_filters, name+"Second_res")
    x = conv2d(name+'r2', x, w2, b2, 1, "SAME")
    
    
    if in_filters != out_filters:
        difference = out_filters - in_filters
        left_pad = difference // 2
        right_pad = difference - left_pad
        identity = tf.pad(in_x, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
        return x + identity
    else:
        return in_x + x

      
def ResNet(_X):
    global n_classes
    w1 = tf.get_variable("FirstW", [5, 5, 3, 30], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("FirstB", [30], initializer=tf.truncated_normal_initializer())
    x = conv2d('conv1', _X, w1, b1, 4, "VALID")
    
    filters_num = [30,50,70,90]
    block_num = [10,15,20,25]
    l_cnt = 1
    for i in range(len(filters_num)):
      for j in range(block_num[i]):
          x = residual_block(x, filters_num[i], filters_num[i], 1, False, 'ResidualBlock%d_%d'%(i,j))
          print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
          l_cnt +=1
          if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
            x = residual_block(x, filters_num[i], filters_num[i+1], 2, True, 'Residualbottom%d_%d'%( i,j))
            print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
            l_cnt +=1

    x = batchnorm(x, filters_num[-1])
    x = activation(x)
    wo, bo=initializer(filters_num[-1], n_classes, "FinalOutput")
    x = conv2d('final', x, wo, bo, 1, "SAME")
    
    
    x = tf.reduce_mean(x, [1,2])
    W = tf.get_variable("FinalW", [n_classes, n_classes], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable("FinalB", [n_classes], initializer=tf.truncated_normal_initializer())
    
    out = tf.matmul(x, W) + b
                            

    return out

#==========================================================================
#=============Reading data in multithreading manner========================
#==========================================================================
def read_labeled_image_list(image_list_file, training_img_dir):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split(' ')
        filename = training_img_dir+filename
        filenames.append(filename)
        labels.append(int(label))
        #print(str(filenames)+"\n")
    return filenames, labels
    
    
def read_images_from_disk(input_queue, size1=128):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    #example = tf.image.decode_jpeg(file_contents, channels=3)
    
    example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    return example, label, fn
    
def setup_inputs(sess, filenames, training_img_dir, image_size=128, crop_size=100, isTest=False, batch_size=50):
    
    # Read each image file
    image_list, label_list = read_labeled_image_list(filenames, training_img_dir)

    images = tf.cast(image_list, tf.string)
    print(str(images)+"\n")
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
        numThr = 4
    else:
        isShuffle = False
        numThr = 1
        
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isShuffle)
    image, y,fn = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        

    image = tf.random_crop(image, [crop_size, crop_size, 3])
    image = tf.cast(image, tf.float32)/255.0
    
    image, y,fn = tf.train.batch([image, y, fn], batch_size=batch_size, capacity=batch_size*3, num_threads=numThr, name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, y, fn, len(label_list)

batch_size = 20 #+ -
display_step = 10
learning_rate = tf.placeholder(tf.float32)      # Learning rate to be fed
lr = 1e-4                          # Learning rate start
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)

# Setup the tensorflow...
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print("Preparing the training & validation data...")
train_data, train_labels, filelist1, glen1 = setup_inputs(sess, "faceMix_FF_DC/train/train.txt", "faceMix_FF_DC/train/", batch_size=batch_size)
val_data, val_labels, filelist2, tlen1 = setup_inputs(sess, "faceMix_FF_DC/test/test.txt", "faceMix_FF_DC/test/", batch_size=batch_size,isTest=True)


max_iter = glen1*100
print("Preparing the training model with learning rate = %.5f..." % (lr))


with tf.variable_scope("ResNet") as scope:
    pred = ResNet(train_data)
    scope.reuse_variables()
    valpred = ResNet(val_data)

with tf.name_scope('Loss_and_Accuracy'):
  cost = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=pred)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
  correct_prediction = tf.equal(tf.argmax(pred, 1), train_labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  top5=tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred, train_labels, 5), tf.float32))
  
  correct_prediction2 = tf.equal(tf.argmax(valpred, 1), val_labels)
  accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
  
  tf.summary.scalar('Loss', cost)
  tf.summary.scalar('Training_Accuracy', accuracy)
  tf.summary.scalar('Top-5_accuracy', top5)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
step = 0
writer = tf.summary.FileWriter("img/", sess.graph)
summaries = tf.summary.merge_all()

print("We are going to train the ImageNet model based on ResNet!!!")
while (step * batch_size) < max_iter:
	epoch1=np.floor((step*batch_size)/glen1)
	if (((step*batch_size)%glen1 < batch_size) & (lr==1e-3) & (epoch1 >2)):
		lr /= 10

	sess.run(optimizer,  feed_dict={learning_rate: lr})

	if (step % display_step == 1):
		# calculate the loss
		loss, acc, top5acc, summaries_string = sess.run([cost, accuracy,top5, summaries])
		print("Iter=%d/epoch=%d, Loss=%.6f, Training Accuracy=%.6f, Top-5 Accuracy=%.6f, lr=%f" % (step*batch_size, epoch1 ,loss, acc, top5acc, lr))
		writer.add_summary(summaries_string, step)
	if (step % (display_step*10) == 1):
		rounds = tlen1 // batch_size
		valacc=[]
		for k in range(rounds):
			a2 = sess.run(accuracy2)
			print("%.6f,"%(a2),end="")
			valacc.append(a2)
		print("\nIter=%d/epoch=%d, Validation Accuracy=%.6f" % (step*batch_size, epoch1 , np.mean(valacc)))
  
	step += 1
print("Optimization Finished!")
#save_path = saver.save(sess, "tf_resnet_model.ckpt")
#print("Model saved in file: %s" % save_path)

exit()
