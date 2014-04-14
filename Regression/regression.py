'''
Created on 15 Feb 2014

Code for MLAP Open Assessment Part 1. 
Linear regression and Logistic regression. 

@author: Y6189686

'''

import csv

from scipy.optimize import fmin_bfgs, minimize
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
import numpy as np
import math
import collections, functools

np.seterr(all='raise')
np.set_printoptions(threshold=np.nan, precision=4)

DEBUG = True
LAMBDA = 0.5

def read_data_file(input_file, calculate_class=False):
    """Read a csv data file. 
    calculate_class determines whether to replace y values with a calculated class value."""
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
            if calculate_class:
                # Calculate the class
                Y[i - 10] = compute_class(float(rows[i][1]), float(rows[i - 1][1]))
                # print X[i - 10], Y[i - 10]
                if i < 20 and DEBUG: 
                    current_price = float(rows[i][1])
                    previous_price = float(rows[i - 1][1])
                    change = current_price - previous_price
                    percentage = (100 * change) / previous_price
                    print i, "Current: ", current_price, " Previous: ", previous_price, " Chage: ", percentage
            else: 
                # Use the price
                Y[i - 10] = rows[i][1]
    if DEBUG: 
        pass
        # print X[0], Y[0], Y[1]
        # print "Last:", X[len(X) - 1], Y[len(Y) - 1]
    return X, Y

def normalize_data(data):
    """"""
    try: 
        # will throw index error if data a vector
        cols = data.shape[1]
        for i in range(0, cols - 1): 
            mean = np.mean(data[:, i])
            sdt = np.std(data[:, i])
            # print mean, sdt
            data[:, i] = (data[:, i] - mean) / sdt
    except IndexError: 
        mean = np.mean(data)
        sdt = np.std(data)
        # print "Array: ", mean, sdt
        data = (data - mean) / sdt
    return data

def mean_squared_loss(theta, x, y):
    """Compute mean squared loss"""
    m, n = np.shape(x)
    hypothesis = np.array([np.dot(a, theta) for a in x])
    loss = hypothesis - y
    mean_squared_loss = np.sum(loss ** 2) / m    
    # print "New mean squared error: ", mean_squared_loss, " | Theta: ", theta
    return mean_squared_loss

def squared_loss(theta, x, y):
    """Compute squared loss"""
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
    """Feature selection"""
    m, n = np.shape(data)  
    
    # return np.array([data[:, i] for i in range(10, len(data-3))])
    # output = np.array([data[:, i] - data[:, i + 1] for i in range(n - 1)])
    # print "output: ", np.shape(output)
    # return output
    slice_size = len(data[0]) / 2
    new_data = np.ones(shape=(len(data), n - slice_size))
    
    new_data[:, 0] = data[: , slice_size ]
    for i in range(slice_size, n - 2): 
        new_data[:, i - slice_size + 1] = data[: , i] - data[: , i + 1]
    
    print new_data[0]
        
    # new_data[:, 1] = data[: , slice_size + 1]
    # new_data[:, 1] = data[: , slice_size] - data[: , slice_size + 1]
    # new_data[:, 2] = data[: , slice_size + 1] - data[: , slice_size + 2]
    # new_data[:, 3] = data[: , 0 ]
    
    return new_data

def slice_and_sum(data, col1, col2):
    data_slice = data[:, range(col1, col2)]
    return np.array([np.sum(row) / len(row) for row in data_slice])

def custom_callback(theta):
    print "Theta: ", theta
    
def find_theta_linear(train_x, train_y, validate_x, validate_y, max_iterations=100):
    m, n = np.shape(train_x)
    theta = np.ones(n)
    theta = fmin_bfgs(mean_squared_loss, theta, args=(train_x, train_y), norm=float('-inf'), maxiter=max_iterations)
    # print theta
    sq_loss = squared_loss(theta, validate_x, validate_y)
    print "validate sq loss: ", sq_loss
    train_msq = mean_squared_loss(theta, train_x, train_y)
    validate_msq = mean_squared_loss(theta, validate_x, validate_y)
    print "Train mean squared loss: ", train_msq
    print "Validate mean squared loss: ", validate_msq
    return theta, validate_msq
    
def linear2_CV():
    x, y = read_data_file('stock_price.csv')
    x = feature_selection_financial_data(x)
    x = normalize_data(x)
    y = normalize_data(y)
    
    fold_1_x = x[0 : len(x) / 2]
    fold_1_y = y[0 : len(y) / 2]
    fold_2_x = x[len(x) / 2 : len(x)]
    fold_2_y = y[len(y) / 2 : len(y)]
    
    fold_1_x_n = normalize_data(x[0 : len(x) / 2])
    fold_1_y_n = normalize_data(y[0 : len(y) / 2])
    fold_2_x_n = normalize_data(x[len(x) / 2 : len(x)])
    fold_2_y_n = normalize_data(y[len(y) / 2 : len(y)])
    
    x_print = np.array(np.arange(len(fold_2_y)))    
    m, n = np.shape(x)    
    print "Shape of data: ", m, " rows, ", n, " cols."
    
    theta1, validate_msq1 = find_theta_linear(fold_1_x_n, fold_1_y_n, fold_2_x, fold_2_y)
    theta2, validate_msq2 = find_theta_linear(fold_2_x_n, fold_2_y_n, fold_1_x, fold_1_y)
    
    print "Average validate mean squared loss: ", (validate_msq1 + validate_msq2) / 2
    
    if DEBUG: 
        line = np.zeros(len(fold_2_y))
        for i in range(n): 
            line += theta1[i] * fold_2_x[:, i]
        plt.plot(x_print, line, 'r-', x_print, fold_2_y, 'o')
        plt.show()
        
###################################################################################
# Logistic regression code
    
def class_probability(theta, x, class_ind):
    top = np.exp(np.dot(x, theta[class_ind]))
    bottom = np.sum([np.exp(np.dot(x, theta[j])) for j in range(len(theta))])
    return  top / bottom 
    
def test_classess(theta, data, y):
    for i in range(50): 
        print "\nData Point ", i
        
        for j in range(len(theta)): 
            print "Class ", j , " with prob ", class_probability(theta, data[i], j), " | Actual: ", y[i]

def accuracy(theta, validate_x, validate_y):
    m, n = np.shape(validate_x)
    k = theta.shape[0]
    success = 0
    classes_count = {}        
    print "\n"
    for i in range(0, m): 
        try: 
            classes_count[validate_y[i]] = classes_count[validate_y[i]] + 1
        except KeyError: 
            classes_count[validate_y[i]] = 1
        calculated_prob = None
        winner = None
        for j in range(0, k): 
            prob = class_probability(theta, validate_x[i], j)
            if prob > calculated_prob: 
                calculated_prob = prob
                winner = j 
        if int(validate_y[i]) == winner: 
            success += 1
    for cl in classes_count.keys():         
        print "Class ", int(cl) , ": ", (float(classes_count[cl]) * 100.) / float(m), "%" 
    accuracy = (float(success) * 100.) / float(m)
    print "\nAccuracy: ", accuracy, "%"
    return accuracy

def find_theta_logistic(train_x, train_y, validate_x, validate_y, max_iterations=30, no_classes=5):
    n = np.shape(train_x)[1]
    theta = np.ones((no_classes, n))
    # print theta
    # theta = minimize(compute_cost, theta, args=(train_x, train_y,))
    theta = fmin_bfgs(compute_cost, theta, args=(train_x, train_y), maxiter=max_iterations)
    theta = theta.reshape((5, n))
    print theta
    test_classess(theta, validate_x, validate_y)
    cost = compute_cost(theta, validate_x, validate_y)
    print "Cost for validate: ", cost
    acc = accuracy(theta, validate_x, validate_y)
    return theta, acc
    
def logistic():
    x, y = read_data_file('stock_price.csv', True)  
    x = feature_selection_financial_data(x)
    print "Data shape: ", y.shape, x.shape
    
    train_x = x[0 : len(x) / 2]
    train_y = y[0 : len(y) / 2]
    validate_x = x[len(x) / 2 : len(x)]
    validate_y = y[len(y) / 2 : len(y)]
    train_x_n = normalize_data(train_x)
    validate_x_n = normalize_data(validate_x)
    
    theta1, accuracy1 = find_theta_logistic(train_x_n, train_y, validate_x, validate_y)
    theta2, accuracy2 = find_theta_logistic(validate_x_n, validate_y, train_x, train_y)
    
    print "Average accuracy: ", (accuracy1 + accuracy2) / 2, "%"
    
def compute_class(current_price, previous_price):
    """Determine the class given the prices of current and past days."""
    change = current_price - previous_price
    percentage = (100 * change) / previous_price
    if abs(percentage) < 5.: 
        price_class = 0
    elif percentage >= 5. and percentage < 10.: 
        price_class = 1
    elif percentage <= -5. and percentage > -10.: 
        price_class = 2 
    elif percentage >= 10.: 
        price_class = 3
    elif percentage <= -10.: 
        price_class = 4
    if DEBUG:
        pass 
        # print "Current: ", current_price, " Previous: ", previous_price, " Chage: ", percentage, "Class: ", price_class
    return price_class

def identity(class1, class2):
    if class1 == class2: 
        return 1
    return 0
    
def compute_cost(theta, X, y):
    global LAMBDA
    theta = theta.reshape((5, np.shape(X)[1]))
    m = X.shape[0] 
    n = theta.shape[0]
    loss = 0.
    # grad = 0.
    for i in range(0, m): 
        current_class = int(y[i])
        probability = class_probability(theta, X[i], current_class)
        loss += -(np.log(probability))
        # for k in range(0, n):
        #    grad += -(np.dot(X[i], (identity(current_class, k) - probability)))
    # summ = np.sum(theta)
    # output = LAMBDA*loss - (1.-LAMBDA)*(summ)
    # print "Output: ", output, "Sum:", summ
    return loss
        
def grad_lr(theta, X, y):
    m = X.shape[0] 
    n = theta.shape[0]
    grad = np.zeros(theta.shape)
    for i in range(0, m): 
        for k in range(0, n):
            prob = class_probability(theta, X[i], k)
            # print prob
            grad[k] = -(np.dot(X[i], (identity(y[i], k) - prob)))
    # print "Grad: ", grad.flatten()
    return grad

if __name__ == '__main__':
    # linear2_CV()
    logistic()
    
# EOF
