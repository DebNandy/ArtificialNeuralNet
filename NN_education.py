# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:44:22 2018

@author: shuva
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:41:03 2018

@author: shuva
"""
import numpy as np
import csv
import sys
def initialize_weights(feature_size):
    w1 = (2*np.random.rand(feature_size,7)-1)/100
    b1 = (2*np.random.rand(1,7)-1)/100
    w2 = (2*np.random.rand(7,1)-1)/100
    b2 = (2*np.random.rand(1,1)-1)/100
    return w1,b1,w2,b2

def get_labels(filename):
    file = open(filename, "r")
    data = file.readlines()
    Output = np.empty((len(data)))
    for i in range(len(data)):
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
            inputarray[i][j] = input[i+1][j]
    return inputarray

def preprocess_input(inputarray):
    #mean = np.mean(inputarray, axis=0)
    #std = np.std(inputarray,axis=0)
    Input_centered = (inputarray)/100
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
    delta1 = a1*(1-a1)*np.dot(w2,delta2)
    return delta1,delta2

def calculate_updates(x,a1,delta1,delta2):
    delw2 = np.dot(a1,np.transpose(delta2))
    delb2 = np.transpose(delta2)
    delw1 = np.dot(x,np.transpose(delta1))
    delb1 = np.transpose(delta1)
    return delw1,delb1,delw2,delb2

def calculate_dev_error(w1,b1,w2,b2):
    DevRawInput = get_input("education_dev.csv")
    DevInput = preprocess_input(DevRawInput)
    DevOutput = make_predictions(DevInput,w1,b1,w2,b2)
    DevTarget = get_labels("education_dev_keys.txt")
    count_samples, count_feature = DevInput.shape
    error = 0
    for i in range(count_samples):
        #print(100*round(DevOutput[i][0],4), DevTarget[i])
        error = error + pow(100*round(DevOutput[i][0],4)- DevTarget[i],2)
    return error/(2*count_samples)

def run_gradient_descent_NN(Input, Output, epochs=5000):
    count_sample,count_feature = Input.shape
    w1,b1,w2,b2 = initialize_weights(count_feature)
    eta = 10
    error_arr = np.empty((epochs,1))
    for i in range(epochs):
        if i>1000:
            eta = 8
        elif i>2000:
            eta = 6
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
            delta1,delta2 = calculate_deltas(a1,a2, y,w2)
            #print(delta1,delta2,delta3)
            delw1,delb1,delw2,delb2 = calculate_updates(x,a1,delta1,delta2)
            totdelw1 = totdelw1+delw1
            totdelb1 = totdelb1+delb1
            totdelw2 = totdelw2+delw2
            totdelb2 = totdelb2+delb2
            #print(a3)
            error = error + (y-a2)*(y-a2)
        error_arr[i]=error
        loss = error/(2)
        print(round(loss[0][0],6))
        #if i%500 == 0 or i == epochs-1 :
         #   print("loss at epoch: ", i, "is: ", round(loss[0][0],4))
          #  dev_error = calculate_dev_error(w1,b1,w2,b2)
           # print("dev error at epoch: ", i, "is: ", round(dev_error,4))
        w1 = w1 + totdelw1*eta/count_sample
        b1 = b1 + totdelb1*eta/count_sample
        w2 = w2 + totdelw2*eta/count_sample
        b2 = b2 + totdelb2*eta/count_sample
        
    return w1,b1,w2,b2, error_arr

def make_predictions(TestInput,w1,b1,w2,b2):
    count_sample, count_feature = TestInput.shape
    output = np.empty((count_sample,1))
    for i in range(count_sample):
        x = np.reshape(TestInput[i],(count_feature,1))
        a1,a2 = calculate_output(x,w1,b1,w2,b2)
        output[i] = a2
    return output

def run_education_NN(edu_train_filename, edu_train_label_filename, max_iter):
    RawInput = get_input(edu_train_filename)
    Input = preprocess_input(RawInput)
    Output = get_labels(edu_train_label_filename)
    ProcessOutput = Output/100
    w1,b1,w2,b2, error_arr = run_gradient_descent_NN(Input,ProcessOutput,epochs=max_iter)
    print("GRADIENT DESCENT TRAINING COMPLETED!")
    print("STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED! NOW PREDICTING.")
    DevRawInput = get_input(edu_dev_filename)
    DevInput = preprocess_input(DevRawInput)
    DevOutput = make_predictions(DevInput,w1,b1,w2,b2)
    count_samples, count_feature = DevInput.shape
    for i in range(count_samples):
        print(round(100*DevOutput[i][0],2))
    return w1,b1,w2,b2


edu_train_filename = sys.argv[1]
edu_train_label_filename = sys.argv[2]
edu_dev_filename = sys.argv[3]

w1,b1,w2,b2 = run_education_NN(edu_train_filename, edu_train_label_filename,7000)


    
    
#w1 = np.array([[-0.26652749,  0.10074024,  0.27946472, -0.12669215, -0.17708352, -0.26026758, -0.14195903],
          #    [-0.41769454,  0.13929772,  0.44508362, -0.18625601, -0.44938827, -0.43006409, -0.44680506],
         #     [-0.26343131,  0.09568584,  0.29703578, -0.10841278, -0.16652555, -0.25401837, -0.15730759],
        #      [-0.42440904,  0.12767939,  0.4398047 , -0.17552516, -0.44317547, -0.42470043, -0.45231245],
       #       [-0.7040475 ,  0.14580916,  0.69534898, -0.28857444, -1.00668923, -0.70256391, -1.04614113]])

#b1 = np.array([[0.23235452,  0.03303393, -0.33753434, -0.0124626 ,  0.94464129,  0.23101326,  1.05952274]])
#w2 = np.array([[-1.14889039],
      # [ 0.50049317],
     #  [ 1.89316556],
    #   [-0.21652577],
   #    [-1.99804598],
  #     [-1.15095909],
 #      [-2.11058828]])
#b2 = np.array([[1.1478278]])
