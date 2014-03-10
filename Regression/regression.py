'''
Created on 15 Feb 2014

@author: Y6189686

'''

import csv
from decimal import Decimal, localcontext
import sys
import time

from scipy.optimize import fmin_bfgs

import datetime as datetime
import matplotlib.pyplot as plt
import numpy as np


# from statsmodels.tools.numdiff import args
np.seterr(all='raise')
np.set_printoptions(threshold=np.nan, precision=4)

LearningRate = 0.0005

def num(number):
    '''Convert a string to a number.'''
    try:
        return int(number)
    except ValueError:
        return float(number)

def read_file(input_file):
    X = []
    Y = []
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            """
            for i in range(len(row) - 1):
                item = num(row[i])
                X.append(item)  
            """
            # print row   
            # np.append(X, num(row[0]))  
            # np.append(Y, float(row[1]))             
            X.append(num(row[0]))   
            Y.append(float(row[1]))
    
    X = np.array(np.arange(0, len(Y)))
    # print X
    # X = np.array(X) 
    Data = np.array([np.ones(len(Y)), X])
    Y = np.array(Y)
    
    return Data.T, X, Y

def read_file2(input_file):
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        # print len(rows)
        X = np.ones(shape=(len(rows) - 10, 21))
        Y = np.zeros(len(rows) - 10)
        # print len(X)
        # print len(rows)
        for i in range(0, len(rows)):
            # print i, rows[i]
            if i < 10: 
                continue
            for n in range(1, 11): 
                X[i - 10][n - 1] = rows[i - n ][0]
            for n in range(11, 21): 
                X[i - 10][n - 1] = rows[i - (n - 10) ][1]
                      
            # if i == 10: 
            #    print X[i - 10]
            
            Y[i - 10] = rows[i][1]
    # print X.shape, Y.shape
    # print Y[0], Y[1]
    # print X[len(X) - 1]
    print X[0], Y[0], Y[1]
    return X, Y

def read_file3(input_file):
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        # print len(rows)
        X = np.ones(shape=(len(rows) - 10, 2))
        Y = np.zeros(len(rows) - 10)
        # print len(X)
        # print len(rows)
        for i in range(0, len(rows)):
            # print i, rows[i]
            if i < 10: 
                continue
            X[i-10][0] = rows[i][2]                      
            # if i == 10: 
            #    print X[i - 10]
            
            Y[i - 10] = rows[i][1]
    # print X.shape, Y.shape
    # print Y[0], Y[1]
    # print X[len(X) - 1]
    print X[0], Y[0], Y[1]
    return X, Y

def feature_selection_financial_data(data):
    new_data = np.ones(shape=(len(data), 4))
    slice_size = len(data[0]) / 2
    new_data[:, 0] = data[: , slice_size ]
    new_data[:, 1] = data[: , slice_size] - data[: , slice_size + 1]
    # new_data[:, 1] = slice_and_sum(data, slice_size, len(data[0]) - 1)
    new_data[:, 2] = slice_and_sum(data, 0, slice_size)
    # new_data[:, 1] = np.log(data[: , slice_size ])
    #new_data[:, 3] = slice_and_sum(data, slice_size, len(data[0]) - 1)
    # new_data[:, 1] = data[: , slice_size + 1 ] **2
    # new_data[:, 1] = data[: , slice_size + 2 ] **2
    # new_data[:, 3] = data[: , slice_size + 1] 
    # new_data[:, range(2, slice_size + 2)] = data[:, range(slice_size, slice_size * 2)]
    return new_data

def slice_and_sum(data, col1, col2):
    data_slice = data[:, range(col1, col2)]
    return np.array([np.sum(row) / len(row) for row in data_slice])


def linear_numpy(input_filename):
    Data, X, Y = read_file(input_filename)    
    w = np.linalg.lstsq(Data, Y)[0]  
    print(w)
    line = w[1] * X + w[0]
    plt.plot(X, line, 'r-', X, Y, 'o')
    plt.show()
    
def linear_numpy2(input_filename):
    x, y = read_file3(input_filename)
    # x = feature_selection_financial_data(x)
    x_print = np.array(np.arange(0, len(y)))
    w = np.linalg.lstsq(x, y)[0]  
    print(w)
    sq_error = loss(w, x, y)
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
    sq_error = loss(w, x, y)
    print sq_error
    line = np.zeros(len(y))
    for i in range(len(x[0])): 
        line += w[i] * x[:, i]
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()

def linear_old(input_filename):
    Data, X, Y = read_file(input_filename)  
    
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, len(Y)), Y, 'ro')

    ax2 = ax1.twinx()
    
    ax2.plot(range(0, len(X)), X, 'bo')
    
    # plt.plot(range(0, len(Y)), Y, 'o', range(0, len(X)), X, 'ro')
    plt.show()

def loss(theta, x, y):
    m, n = np.shape(x)
    # hypothesis = np.dot(x, theta)
    hypothesis = np.array([np.dot(a, theta) for a in x])
    loss = hypothesis - y
    sq_loss = np.sum(loss ** 2) / (m)
    # cost = np.sum(loss ** 2) / (2 * m)
    # gradient = np.dot(x.T, loss)
    # print sq_loss, gradient
    
    return sq_loss

def grad(theta, x, y):
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    gradient = np.dot(x.T, loss)
    return gradient

def linear():
    x, x_print, y = read_file('stock_price.csv')
    m, n = np.shape(x)
    print m, n
    max_iterations = 4000
    theta = np.ones(n)
    theta = fmin_bfgs(loss, theta, args=(x, y), maxiter=max_iterations)
    line = theta[1] * x_print + theta[0]  # regression line    
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()
    print theta
    
def linear2():
    x, y = read_file2('stock_price.csv')
    x_print = np.array(np.arange(0, len(y)))
    m, n = np.shape(x)
    print m, n
    max_iterations = 200
    theta = np.ones(n)
    print theta
    theta = fmin_bfgs(loss, theta, args=(x, y), maxiter=max_iterations)
    print theta
    line = np.zeros(len(y))
    # volume = np.zeros(len(y))
    # line = theta[len(theta) - 1] + np.array([])
    for i in range(n): 
        line += theta[i] * x[:, i]
    # line = theta[1] * x[:,1] + theta[0] *  x[:,0] # regression line    
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()
    
def linear2_CV():
    x, y = read_file2('stock_price.csv')
    x = feature_selection_financial_data(x)
    train_x = x[0 : len(x) / 4]
    train_y = y[0 : len(y) / 4]
    validate_x = x[len(x) / 4 : len(x)]
    validate_y = y[len(y) / 4 : len(y)]
    """
    train_x = np.array([x[n] for n in range(0, len(x), 2)])
    train_y = np.array([y[n] for n in range(0, len(y), 2)])
    validate_x = np.array([x[n] for n in range(1, len(x), 2)])
    validate_y = np.array([y[n] for n in range(1, len(y), 2)])
    """
    x_print = np.array(np.arange(len(validate_y)))
    print x[0]
    m, n = np.shape(x)
    print m, n
    max_iterations = 50
    theta = np.ones(n)
    print theta
    theta = fmin_bfgs(loss, theta, args=(train_x, train_y), maxiter=max_iterations)
    print theta
    sq_loss = loss(theta, validate_x, validate_y)
    print sq_loss
    line = np.zeros(len(validate_y))
    # volume = np.zeros(len(y))
    # line = theta[len(theta) - 1] + np.array([])
    for i in range(n): 
        line += theta[i] * validate_x[:, i]
    plt.plot(x_print, line, 'r-', x_print, validate_y, 'o')
    # plt.plot(validate_x[:, 0], line, 'r-', validate_x[:, 0], validate_y, 'o')
    plt.show()
    
def predict_class(theta, x):
    m, n = x.shape
    p = np.zeros(shape=(m, 1))
 
    h = sigmoid(x.dot(theta))
 
    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
 
    return p

def sigmoid(X):
    '''Compute the sigmoid function '''
    #d = zeros(shape=(X.shape))
 
    den = 1.0 + np.exp (-1.0 * X)
 
    d = 1.0 / den
 
    return d

if __name__ == '__main__':

    # read_file2('stock_price.csv')
    # linear()
    # linear2()
    linear2_CV()
    #linear_numpy2('stock_price_test2.csv')
    
    
    
    
    
