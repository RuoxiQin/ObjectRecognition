#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The neural net classifier
"""

import numpy as np
import tensorflow as tf
import os
from caffe_classes import class_names
from gen_train import input_func
from gen_test import input_func_test
import timeit
import sys
import pickle

PICTURE_SIZE = 227
LEARN_RATE = 0.0001
TRAIN_STEPS = 50
CLASS_NUM = 2
TRAIN = 0
EVAL = 1
PREDICT = 2
test_file_name = "test_data.p"

class inputs:
    features, labels = input_func()
    features["objects"] = np.array(features["objects"]).astype(np.float32)
    features["scenes"] = np.array(features["scenes"]).astype(np.float32)
    labels = np.array(labels).astype(np.int32)
    # Shuffle the data
    index = np.arange(labels.shape[0])
    np.random.shuffle(index)
    features["objects"] = features["objects"][index]
    features["scenes"] = features["scenes"][index]
    labels = labels[index]

def dummy_input_fn():
    return inputs.features, inputs.labels
        

def train_input_fn():
    """
    The input function of our neural net
    """
    features, labels = input_func()
    features["objects"] = np.array(features["objects"]).astype(np.float32)
    features["scenes"] = np.array(features["scenes"]).astype(np.float32)
    labels = np.array(labels).astype(np.int32)
    # Shuffle the data
    index = np.arange(labels.shape[0])
    np.random.shuffle(index)
    features["objects"] = features["objects"][index]
    features["scenes"] = features["scenes"][index]
    labels = labels[index]
    return features, labels
    

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
    conv1_in = conv(input_layer, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, 
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


def model_fn(features, labels, mode):
    """
    The model function for tf.Estimator
    """
    # Object Input layer
    input_object = tf.reshape(\
        features["objects"], [-1, PICTURE_SIZE, PICTURE_SIZE, 3])
    # Scene Input layer
    input_scene = tf.reshape(\
        features["scenes"], [-1, PICTURE_SIZE, PICTURE_SIZE, 3])
    # Use the pretrained AlexNet weight to form the CNN
    last_cnn_object = generate_CNN(input_object)
    last_cnn_scene = generate_CNN(input_scene)
    # Concatenate layer
    concat_layer = tf.concat([last_cnn_object, last_cnn_scene], axis=1)
    # Dense layer1
    dense1 = tf.layers.dense(inputs=concat_layer, units=4096, \
        activation=tf.nn.relu)
    # Dropout layer1
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, \
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Dense layer2
    dense2 = tf.layers.dense(inputs=dropout1, units=4096, \
        activation=tf.nn.relu)
    # Dropout layer2
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, \
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Logits layer
    logits = tf.layers.dense(inputs=dropout1, units=CLASS_NUM)

    # prediction result in PREDICT and EVAL phases
    predictions = {
        # Class id
        "classes": tf.argmax(input=logits, axis=1),
        # Probabilities
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss for TRAIN and EVAL
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE)
        train_op = optimizer.minimize(\
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(\
            mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(\
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(\
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(status, features=None, labels=None):
    """
    Load the training and testing data
    """
    # Create the Estimator
    classifier = tf.estimator.Estimator(\
        model_fn=model_fn, model_dir="./tmp")

    # Setup logging hook for prediction
    tf.logging.set_verbosity(tf.logging.INFO)
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    '''
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=features,
        y=labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)
    '''
    if status == TRAIN:
        classifier.train(
            input_fn=train_input_fn,
            steps=TRAIN_STEPS,
            hooks=[logging_hook])
    elif status == EVAL and features is not None and labels is not None:
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=features,
            y=labels,
            num_epochs=1,
            shuffle=False)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        return eval_results
    elif status == PREDICT and features is not None:
        features["objects"] = \
            np.array(features["objects"]).astype(np.float32)
        features["scenes"] = \
            np.array(features["scenes"]).astype(np.float32)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=features,
            num_epochs=1,
            shuffle=False)
        predict_results = classifier.predict(\
            input_fn=predict_input_fn)
        return predict_results

class Detector:
    """
    The Detector
    """
    def __init__(self, model_dir):
        # Create the Estimator
        self.classifier = tf.estimator.Estimator(\
            model_fn=model_fn, model_dir=model_dir)

    def train(self):
        """
        Train the model
        """
        # Setup logging hook for prediction
        tf.logging.set_verbosity(tf.logging.INFO)
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
        self.classifier.train(
            input_fn=train_input_fn,
            steps=TRAIN_STEPS,
            hooks=[])

    def evaluate(self, features, labels):
        """
        Evaluate the model
        """
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=features,
            y=labels,
            num_epochs=1,
            shuffle=False)
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        return eval_results

    def predict(self, features):
        """
        Classify whether the object exists in the scene
        """
        features["objects"] = np.array(features["objects"]).astype(np.float32)
        features["scenes"] = np.array(features["scenes"]).astype(np.float32)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=features,
            num_epochs=1,
            shuffle=False)
        predict_iterator = self.classifier.predict(\
            input_fn=predict_input_fn)
        prediction = [pre for pre in predict_iterator]
        return prediction


if __name__ == "__main__":
    # Generate a fixed testing data
    if not os.path.isfile(test_file_name):
        print("No previous testing data!")
        test_features, test_labels = input_func_test()
        test_features["objects"] = \
            np.array(test_features["objects"]).astype(np.float32)
        test_features["scenes"] = \
            np.array(test_features["scenes"]).astype(np.float32)
        test_labels = np.array(test_labels).astype(np.int32)
        test_data = {"features": test_features, "labels": test_labels}
        pickle.dump(test_data, open(test_file_name, "wb"))
    else:
        print("Load saved testing data!")
        test_data = pickle.load(open(test_file_name, "rb"))
        test_features = test_data["features"]
        test_labels = test_data["labels"]
    # Start training
    record_file_path = "./tmp/original/accuracy_record.txt"
    detector = Detector("./tmp/original")
    '''
    f = open(record_file_path, "a+")
    f.write("Start a new training...\n")
    f.close()
    for i in range(100):
        detector.train()
        print("Training step %d:" % i)
        evaluation_result = detector.evaluate(test_features, test_labels)
        print(evaluation_result)
        f = open(record_file_path, "a+")
        f.write("Training step %d:\n" % i)
        f.write(str(evaluation_result))
        f.write("\n")
        f.close()
    print("Done!")
    '''
    result = detector.predict(test_features)
    print(result)