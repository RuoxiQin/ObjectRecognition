import numpy as np
import tensorflow as tf
import os
from caffe_classes import class_names
from gen_train import input_func
import timeit

PICTURE_SIZE = 227
LEARN_RATE = 0.001
TRAIN_STEPS = 301

def round_01(x):
    return 0 if x < 0.5 else 1
round_01 = np.vectorize(round_01)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", 
    group=1):
    '''
    The function to generate the pretrained ALEXNET
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], 
        padding=padding)
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, 
            kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), 
        [-1]+conv.get_shape().as_list()[1:])

def generate_CNN(input_layer):
    """
    The function to generate AlexNet CNN
    """
    # CNN for the object workflow
    # Load weighs of Alex Net
    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0], trainable=False)
    conv1b = tf.Variable(net_data["conv1"][1], trainable=False)
    conv1_in = conv(input_object, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, 
        padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius,
        alpha=alpha, beta=beta, bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], 
        strides=[1, s_h, s_w, 1], padding=padding)

    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0], trainable=False)
    conv2b = tf.Variable(net_data["conv2"][1], trainable=False)
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, 
        padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius,
        alpha=alpha, beta=beta, bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], 
        strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0], trainable=False)
    conv3b = tf.Variable(net_data["conv3"][1], trainable=False)
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, 
        padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0], trainable=False)
    conv4b = tf.Variable(net_data["conv4"][1], trainable=False)
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, 
        padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0], trainable=False)
    conv5b = tf.Variable(net_data["conv5"][1], trainable=False)
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, 
    padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], 
        strides=[1, s_h, s_w, 1], padding=padding)

    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0], trainable=False)
    fc6b = tf.Variable(net_data["fc6"][1], trainable=False)
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, 
        [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    return fc6


mode_is_train = tf.placeholder(tf.bool)
# Input layers
input_object = tf.placeholder(tf.float32, [None, PICTURE_SIZE, PICTURE_SIZE, 3])
input_scene = tf.placeholder(tf.float32, [None, PICTURE_SIZE, PICTURE_SIZE, 3])
ground_truth = tf.placeholder(tf.float32, [None, 1])
# Use the pretrained AlexNet weight to form the CNN
last_cnn_object = generate_CNN(input_object)
last_cnn_scene = generate_CNN(input_scene)
# Concatenate two layer
concat_layer = tf.concat([last_cnn_object, last_cnn_scene], axis=1)
# Add 1 fully connected hidden layer after the cnn
dense1 = tf.layers.dense(inputs=concat_layer, units=8192, activation=tf.nn.relu)
dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training = mode_is_train)
logits = tf.layers.dense(inputs=dropout1, units=1)
sig_logits = tf.sigmoid(logits)
# define loss function
loss = tf.losses.mean_squared_error(predictions=sig_logits, labels=ground_truth)
# add optimizer, this a symbolic ops to do gradient descent
train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)
# add evaluate ops, this is used to evaluate the model
predicted_class = tf.greater(logits, 0.5)
correct = tf.equal(predicted_class, tf.equal(ground_truth, 1.0))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
# initializer
init = tf.global_variables_initializer()
# Add ops to save and restore the trained fully connected layer
saver = tf.train.Saver()
# Train and test the result
with tf.Session() as sess:
    sess.run(init)

    # Training phase
    print("In training phase:")
    for step in range(TRAIN_STEPS):
        print("Step %d training" % step)
        start_time = timeit.default_timer()
        features, labels = input_func()
        features["objects"] = np.array(features["objects"]).astype(np.float32)
        features["scenes"] = np.array(features["scenes"]).astype(np.float32)
        labels = np.array(labels).reshape((-1, 1)).astype(np.float32)
        train_step.run(feed_dict=
            {input_object: features["objects"], 
            input_scene: features["scenes"], 
            ground_truth: labels, 
            mode_is_train: True})
        # Save the trained weights
        end_time = timeit.default_timer()
        print("The training time for this epcho is: %d" % \
            (end_time - start_time))

        # Save the model and Print the accuracy periodically
        if step % 5 == 0:
            # load the data
            #saver.restore(sess, './tmp/model_whole.ckpt')
            # save the trained weights
            save_path = saver.save(sess, "./tmp/model_whole.ckpt")
            save_end_time = timeit.default_timer()
            print("The saving time is: %d" % \
                (save_end_time - end_time))
            print("Step %d trained weight has saved" % step)
            features, labels = input_func()
            features["objects"] = \
                np.array(features["objects"]).astype(np.float32)
            features["scenes"] = np.array(features["scenes"]).astype(np.float32)
            labels = np.array(labels).reshape((-1, 1)).astype(np.float32)
            print('the logit is:')
            print(sess.run(logits, feed_dict=
                {input_object: features["objects"], 
                input_scene: features["scenes"], 
                ground_truth:labels, mode_is_train: False}))
            print('the ground truth is:')
            print(sess.run(ground_truth, feed_dict=
                {input_object: features["objects"], 
                input_scene: features["scenes"], 
                ground_truth:labels, mode_is_train: False}))
            print("The loss is:")
            print(loss.eval(feed_dict=
                {input_object: features["objects"], 
                input_scene: features["scenes"], 
                ground_truth:labels, mode_is_train: False}))
            labels = round_01(labels).astype(np.float32)
            print("The accuracy is:")
            train_accuracy = accuracy.eval(feed_dict=
                {input_object: features["objects"], 
                input_scene: features["scenes"], 
                ground_truth:labels, mode_is_train: False})
            print("Step %d, training accuracy of this batch is %f" % (step,   
                train_accuracy))
            test_end_time = timeit.default_timer()
            print("The testing time is %d" % \
                (test_end_time - save_end_time))

