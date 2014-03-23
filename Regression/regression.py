'''
Created on 15 Feb 2014

Code for MLAP Open Assessment Part 1. 
Linear regression and Logistic regression. 

@author: Y6189686

'''

import csv

from scipy.optimize import fmin_bfgs
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
import numpy as np
import math


np.seterr(all='raise')
np.set_printoptions(threshold=np.nan, precision=4)

def read_file2(input_file):
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        X = np.ones(shape=(len(rows) - 10, 21))
        Y = np.zeros(len(rows) - 10)
        for i in range(0, len(rows)):
            # print i, rows[i]
            if i < 10: 
                continue
            for n in range(1, 11): 
                X[i - 10][n - 1] = rows[i - n ][0]
            for n in range(11, 21): 
                X[i - 10][n - 1] = rows[i - (n - 10) ][1]            
            Y[i - 10] = rows[i][1]
    print X[0], Y[0], Y[1]
    print "Last:", X[len(X) - 1], Y[len(Y) - 1]
    return X, Y

def read_file4(input_file):
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        X = np.ones(shape=(len(rows) - 10, 21))
        Y = np.zeros(len(rows) - 10)
        for i in range(0, len(rows)):
            # print i, rows[i]
            if i < 10: 
                continue
            for n in range(1, 11): 
                X[i - 10][n - 1] = rows[i - n ][0]
            for n in range(11, 21): 
                X[i - 10][n - 1] = rows[i - (n - 10) ][1]    
            # X[i - 10][20] = price(float(rows[i - 9][1]), float(rows[i - 10][1]))
            # Y[i - 10] = rows[i][1]
            Y[i - 10] = price(float(rows[i - 9][1]), float(rows[i - 10][1]))
    print X[0], Y[0], Y[1]
    print "Last:", X[len(X) - 1], Y[len(Y) - 1]
    return X, Y

def read_file3(input_file):
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        X = np.ones(shape=(len(rows) - 10, 2))
        Y = np.zeros(len(rows) - 10)
        for i in range(0, len(rows)):
            if i < 10: 
                continue
            X[i - 10][0] = rows[i][2]                  
            Y[i - 10] = rows[i][1]
    print X[0], Y[0], Y[1]    
    return X, Y

def normalize_data(data):
    for i in range(0, data.shape[1]-1): 
        mean = np.mean(data[:, i])
        sdt = np.std(data[:, i])
        print mean, sdt
        data[:, i] = (data[:, i] - mean) / sdt
    print "Normal: ", data[0]
    return data

def mean_squared_loss(theta, x, y):
    m, n = np.shape(x)
    hypothesis = np.array([np.dot(a, theta) for a in x])
    loss = hypothesis - y
    mean_squared_loss = np.sum(loss ** 2) / m    
    # print "New mean squared error: ", mean_squared_loss, " | Theta: ", theta
    return mean_squared_loss

def squared_loss(theta, x, y):
    m, n = np.shape(x)
    hypothesis = np.array([np.dot(a, theta) for a in x])
    loss = hypothesis - y
    squared_loss = np.sum(loss ** 2)
    # print "New mean squared error: ", mean_squared_loss, " | Theta: ", theta
    return squared_loss

def grad(theta, x, y):
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    gradient = np.dot(x.T, loss)
    return gradient

def feature_selection_financial_data(data):  
    m, n = np.shape(data)  
    # return np.array([data[:, i] for i in range(10, len(data-3))])
    output = np.array([data[:, i] - data[:, i + 1] for i in range(n - 1)])
    print "output: ", np.shape(output)
    # return output
    new_data = np.ones(shape=(len(data), 3))
    slice_size = len(data[0]) / 2
    
    new_data[:, 0] = data[: , slice_size ]
    new_data[:, 1] = data[: , slice_size] - data[: , slice_size + 1]
    # new_data[:, 2] = slice_and_sum(data, slice_size, len(data[0]) - 1)   
    
    return new_data

def slice_and_sum(data, col1, col2):
    data_slice = data[:, range(col1, col2)]
    return np.array([np.sum(row) / len(row) for row in data_slice])
    
def linear_numpy2(input_filename):
    x, y = read_file3(input_filename)
    # x = feature_selection_financial_data(x)
    x_print = np.array(np.arange(0, len(y)))
    w = np.linalg.lstsq(x, y)[0]  
    print(w)
    sq_error = mean_squared_loss(w, x, y)
    print sq_error
    line = np.zeros(len(y))
    for i in range(len(x[0])): 
        line += w[i] * x[:, i]
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()
    
def linear_numpy2_cv(input_filename):
    x, y = read_file3(input_filename)
    # x = feature_selection_financial_data(x)
    
    train_x = x[0 : len(x) / 2]
    train_y = y[0 : len(y) / 2]
    validate_x = x[len(x) / 2 : len(x)]
    validate_y = y[len(y) / 2 : len(y)]
    
    x_print = np.array(np.arange(0, len(y)))
    w = np.linalg.lstsq(x, y)[0]  
    print(w)
    sq_error = mean_squared_loss(w, x, y)
    print sq_error
    line = np.zeros(len(y))
    for i in range(len(x[0])): 
        line += w[i] * x[:, i]
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()
    
def linear2():
    x, y = read_file2('stock_price.csv')
    x_print = np.array(np.arange(0, len(y)))
    m, n = np.shape(x)
    print m, n
    max_iterations = 200
    theta = np.ones(n)
    print theta
    theta = fmin_bfgs(mean_squared_loss, theta, args=(x, y), maxiter=max_iterations)
    print theta
    line = np.zeros(len(y))
    for i in range(n): 
        line += theta[i] * x[:, i]
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()
    
def custom_callback(theta):
    print "Theta: ", theta
    
def linear2_CV():
    x, y = read_file2('stock_price.csv')
    x = feature_selection_financial_data(x)
    x = normalize_data(x)
    train_x = x[0 : len(x) / 2]
    train_y = y[0 : len(y) / 2]
    validate_x = x[len(x) / 2 : len(x)]
    validate_y = y[len(y) / 2 : len(y)]
    """
    train_x = np.array([x[n] for n in range(0, len(x), 2)])
    train_y = np.array([y[n] for n in range(0, len(y), 2)])
    validate_x = np.array([x[n] for n in range(1, len(x), 2)])
    validate_y = np.array([y[n] for n in range(1, len(y), 2)])
    """
    x_print = np.array(np.arange(len(validate_y)))
    # print x[0]
    print train_x[0]
    m, n = np.shape(x)
    print "Shape of data: ", m, " rows, ", n, " cols."
    max_iterations = 400
    theta = np.ones(n)
    # print theta
    theta = fmin_bfgs(mean_squared_loss, theta, args=(train_x, train_y), norm=float('-inf'), maxiter=max_iterations)
    print theta
    sq_loss = squared_loss(theta, validate_x, validate_y)
    print "validate sq loss: ", sq_loss
    train_msq = mean_squared_loss(theta, train_x, train_y)
    validate_msq = mean_squared_loss(theta, validate_x, validate_y)
    print "Train mean squared loss: ", train_msq
    print "Validate mean squared loss: ", validate_msq
    line = np.zeros(len(validate_y))
    # volume = np.zeros(len(y))
    # line = theta[len(theta) - 1] + np.array([])
    for i in range(n): 
        line += theta[i] * validate_x[:, i]
    plt.plot(x_print, line, 'r-', x_print, validate_y, 'o')
    # plt.plot(validate_x[:, 0], line, 'r-', validate_x[:, 0], validate_y, 'o')
    plt.show()
    
def is_class(theta, x, class_ind):
    # print theta, x, class_ind
    top = np.exp(np.dot(x, theta[class_ind]))
    bottom = np.sum([np.exp(np.dot(x, theta[j])) for j in range(len(theta))])
    # print "tb: ", top, bottom
    
    # if bottom == 0: 
    #    return 0
    return  top / bottom 
    
def test_classess(theta, data, y):
    for i in range(50): 
        print "\nData Point ", i
        
        for j in range(len(theta)): 
            print "Class ", j , " with prob ", is_class(theta, data[i], j), " | Actual: ", y[i]

def accuracy(theta, x, y):
    m, n = np.shape(x)
    k = theta.shape[0]
    success = 0
    class_0 = 0
    for i in range(0, m): 
        if(y[i]) == 0: 
            class_0 += 1
    print "\nClass 0: ", (class_0 * 100) / m, "%"
    for i in range(0, m): 
        calculated_prob = None
        winner = None
        for j in range(0, k): 
            prob = is_class(theta, x[i], j)
            if prob > calculated_prob: 
                calculated_prob = prob
                winner = j 
        if int(y[i]) == winner: 
            success += 1
    print "\nAccuracy: ", (success * 100) / m, "%"
            

def logistic():
    x, y = read_file4('stock_price.csv')  
    x = feature_selection_financial_data(x)
    x = normalize_data(x)
    train_x = x[0 : len(x) / 2]
    train_y = y[0 : len(y) / 2]
    validate_x = x[len(x) / 2 : len(x)]
    validate_y = y[len(y) / 2 : len(y)]
    print y.shape, x.shape
    m, n = np.shape(x)
    max_iterations = 100
    theta = np.ones((5, n))
    print theta.shape
    theta = fmin_bfgs(compute_cost2, theta, args=(train_x, train_y), maxiter=max_iterations)
    theta = theta.reshape((5, np.shape(x)[1]))
    print theta
    test_classess(theta, validate_x, validate_y)
    accuracy(theta, validate_x, validate_y)
    
def price(current_price, previous_price):
    change = current_price - previous_price
    percentage = (100 * change) / previous_price
    # print "Current: ", current_price, " Previous: ", previous_price, " Chage: ", percentage
    if abs(percentage) < 5.: 
        return 0
    elif percentage >= 5. and percentage < 10.: 
        return 1
    elif percentage <= -5. and percentage > -10.: 
        return 2 
    elif percentage >= 10.: 
        return 3
    elif percentage <= -10.: 
        return 4
    
def compute_cost(theta, X, y): 
    theta = theta.reshape((5, np.shape(X)[1]))
    # print theta
    m = X.shape[0] 
    n = theta.shape[0]
    calculated_log = np.sum(np.array([log_sum_exp(np.dot(X, theta[i])) for i in range(0, n)]))   
    loss = 0
    for i in range(0, n): 
        loss += np.sum(np.dot(X, theta[i]) - calculated_log) 
    # loss = np.sum((np.dot(X, theta[i]) - calculated_log) for i in range(n))
    print -loss, theta.flatten()
    return -loss
    # return J[0][0]

def identity(class1, class2):
    if class1 == class2: 
        return 1
    return 0

def compute_cost3(theta, X, y):
    theta = theta.reshape((5, np.shape(X)[1]))
    # print theta
    m = X.shape[0] 
    n = theta.shape[0]
    loss = 0   
    # theta_x = np.dot(theta, X) 
    theta_x = np.array(np.dot(theta, X[i]) for i in range(0, m))
    print theta_x
    hypothesis = np.exp(theta_x)      
    probabilities = hypothesis / np.sum(hypothesis, axis=0)
    lse = np.log(probabilities)
    
def compute_cost2(theta, X, y):
    theta = theta.reshape((5, np.shape(X)[1]))
    # print theta
    m = X.shape[0] 
    n = theta.shape[0]
    loss = 0    
    grad = np.zeros((m, n))
    for i in range(0, m): 
        lse = logsumexp([np.dot(theta[j], X[i]) for j in range(0, n)])
        theta_x = np.dot(theta[y[i]], X[i])
        loss += -(theta_x - lse)
        # for k in range(0, n):
        #    prob = is_class(theta, X[i], k)
        #    print prob
        #    grad[i][k] = -np.sum(np.dot(X[i], (identity(y[i], k) - prob)))
    print "Output: ", loss
    return loss

def log_sum_exp(Vector):
    # first find max of the sequence
    Max = Vector[0]
    for LP in Vector:
        if (LP > Max): 
            Max = LP
    Sum = 0.0
    for LP in Vector:
        Sum += np.exp(LP - Max)    
    jazz = (Max + np.log(Sum))
        
    return jazz

def predict_class(theta, target, x):
    # m, n = x.shape
    # p = np.zeros(shape=(m, 1))
    
    
    theta_x = np.dot(theta, x)
    h = np.exp(theta_x)
    probabilities = h / np.sum(h, axis=0)
 
    predictions = np.zeros((len(x), 1))
    predictions[:, 0] = np.argmax(probabilities, axis=0)
    return predictions
    """
    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
 
    return p"""

def sigmoid(X):
    '''Compute the sigmoid function '''
    # d = zeros(shape=(X.shape))
 
    den = 1.0 + np.exp (-1.0 * X)
 
    d = 1.0 / den
 
    return d

if __name__ == '__main__':

    # linear2()
    #linear2_CV()
    logistic()
    # dummy_test()
    # linear_numpy2('stock_price_test2.csv')
    
# EOF
