# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:41:03 2018

@author: shuva
"""
import numpy as np
import csv
import sys

def initialize_weights(feature_size):
    w1 = (2*np.random.rand(feature_size,5)-1)/100
    b1 = (2*np.random.rand(1,5)-1)/100
    w2 = (2*np.random.rand(5,1)-1)/100
    b2 = (2*np.random.rand(1,1)-1)/100
    return w1,b1,w2,b2

def get_labels(filename):
    file = open(filename, "r")
    data = file.readlines()
    Output = np.empty((len(data)))
    for i in range(len(data)):
        if data[i] == 'yes\n' :
            Output[i] = 1
        elif data[i] == 'no\n' :
            Output[i] = 0
        else :
            Output[i] = data[i]
    file.close()
    return Output

def get_input(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        input = list(reader)
    count_sample = len(input)-1
    count_feature = len(input[0])
    inputarray = np.empty((count_sample,count_feature))
    for i in range(count_sample):
        for j in range(count_feature):
            if input[i+1][j] == 'yes':
                input[i+1][j] = 1
            elif input[i+1][j] == 'no' :
                input[i+1][j] = 0
            inputarray[i][j] = input[i+1][j]
    return inputarray

def preprocess_input(inputarray):
    mean = np.mean(inputarray, axis=0)
    std = np.std(inputarray,axis=0)
    adjustment = np.array([1900,0,0,0])
    normalizer = np.array([100,7,1,1])
    Input_centered = (inputarray-mean)/std
    return Input_centered

def sigmoid(x):
    val = 1/(1+np.exp(-x))
    return val
    
def calculate_output(x,w1,b1,w2,b2):
    z1 = np.dot(np.transpose(w1),x) + np.transpose(b1)
    a1 = sigmoid(z1)
    z2 = np.dot(np.transpose(w2),a1) + np.transpose(b2)
    a2 = sigmoid(z2)
    return a1,a2

def calculate_deltas(a1,a2,y,w2):
    delta2 = a2*(1-a2)*(y-a2)
    delta1 = a1*(1-a1)*w2*delta2
    return delta1,delta2

def calculate_updates(x,a1,delta1,delta2):
    delw2 = np.dot(a1,np.transpose(delta2))
    delb2 = np.transpose(delta2)
    delw1 = np.dot(x,np.transpose(delta1))
    delb1 = np.transpose(delta1)
    return delw1,delb1,delw2,delb2

def run_gradient_descent_NN(Input, Output, epochs=5000):
    count_sample,count_feature = Input.shape
    w1,b1,w2,b2 = initialize_weights(count_feature)
    eta = 2.5
    error_arr = np.empty((5000,1))
    for i in range(epochs):
        if i>1000:
            eta = 2
        elif i>2000:
            eta = 1.5
        elif i>3000:
            eta = 1
        totdelw1 = np.zeros(w1.shape)
        totdelb1 = np.zeros(b1.shape)
        totdelw2 = np.zeros(w2.shape)
        totdelb2 = np.zeros(b2.shape)
        error = 0
        loss = 0
        for j in range(count_sample):
            x = np.reshape(Input[j],(count_feature,1))
            y = np.reshape(Output[j],(1,1))
            a1,a2 = calculate_output(x,w1,b1,w2,b2)
            delta1,delta2 = calculate_deltas(a1,a2,y,w2)
            delw1,delb1,delw2,delb2 = calculate_updates(x,a1,delta1,delta2)
            totdelw1 = totdelw1+delw1
            totdelb1 = totdelb1+delb1
            totdelw2 = totdelw2+delw2
            totdelb2 = totdelb2+delb2
            error = error + (y-a2)*(y-a2)
        error_arr[i]=error
        loss = error/(2)
        w1 = w1 + totdelw1*eta/count_sample
        b1 = b1 + totdelb1*eta/count_sample
        w2 = w2 + totdelw2*eta/count_sample
        b2 = b2 + totdelb2*eta/count_sample
        #if i%500 == 0 or i == epochs-1:
        print(round(loss[0][0],6))
    return w1,b1,w2,b2,error_arr

def make_predictions(TestInput,w1,b1,w2,b2):
    count_sample, count_feature = TestInput.shape
    output = np.empty((count_sample,1))
    for i in range(count_sample):
        x = np.reshape(TestInput[i],(count_feature,1))
        a1,a2 = calculate_output(x,w1,b1,w2,b2)
        output[i] = a2
    return output

def calculate_dev_error(w1,b1,w2,b2):
    DevRawInput = get_input(music_dev_filename)
    DevInput = preprocess_input(DevRawInput)
    DevOutput = make_predictions(DevInput,w1,b1,w2,b2)
    count_samples, count_feature = DevInput.shape
    DevTarget = get_labels("music_dev_keys.txt")
    error = 0
    for i in range(count_samples):
        error = error + pow(DevOutput[i][0]-DevTarget[i],2)
    return error/2

def run_music_NN(music_train_filename, music_train_label_filename, music_dev_filename,max_iter):
    RawInput = get_input(music_train_filename)
    Input = preprocess_input(RawInput)
    Output = get_labels(music_train_label_filename)
    w1,b1,w2,b2, error_arr = run_gradient_descent_NN(Input,Output, epochs=max_iter)
    print("GRADIENT DESCENT TRAINING COMPLETED!")
    
    print("STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED! NOW PREDICTING.")
    DevRawInput = get_input(music_dev_filename)
    DevInput = preprocess_input(DevRawInput)
    DevOutput = make_predictions(DevInput,w1,b1,w2,b2)
    count_samples, count_feature = DevInput.shape
    for i in range(count_samples):
        if DevOutput[i] >= 0.5:
            print('yes')
        else :
            print('no')


music_train_filename = sys.argv[1]
music_train_label_filename = sys.argv[2]
music_dev_filename = sys.argv[3]

run_music_NN(music_train_filename,music_train_label_filename, music_dev_filename,3000)

#print(get_input(music_train_filename))
#Dev = np.array([1,2.3])
#print(round(Dev[0]))



    
    
    
    
    
