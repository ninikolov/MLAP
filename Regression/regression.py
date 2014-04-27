'''
Created on 15 Feb 2014

Code for MLAP Open Assessment, Part 1. 
Linear regression and Logistic regression. 

@author: Y6189686

'''

import csv
from sklearn import linear_model
import copy, random
import time

from scipy.optimize import fmin_bfgs, minimize

import matplotlib.pyplot as plt
from pylab import scatter, show, legend , xlabel, ylabel
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OutputCodeClassifier

# Raise calculation errors as exceptions - such as floating point error. 
np.seterr(all='raise')
np.set_printoptions(threshold=np.nan, precision=4)

DEBUG = True
LAMBDA = 0.5

def read_data_file(input_file, calculate_class=False):
    """Read a csv data file. Used for both linear and logistic regression. 
    calculate_class determines whether to replace y values with a calculated class value."""
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        X = np.ones(shape=(len(rows) - 10, 21))
        if calculate_class: 
            Y = np.zeros(len(rows) - 10, dtype=np.uint8)
        else: 
            Y = np.zeros(len(rows) - 10)
        for i in range(0, len(rows)):
            if i < 10: 
                continue
            for n in range(1, 11):  # 10 previous stock volumes
                X[i - 10][n - 1] = rows[i - n ][0]
            for n in range(11, 21):  # 10 previous prices
                X[i - 10][n - 1] = rows[i - (n - 10) ][1]
            if calculate_class:  # Calculate the class from the prices                
                Y[i - 10] = compute_class(float(rows[i][1]), float(rows[i - 1][1]))
            else:  # Use the price                
                Y[i - 10] = rows[i][1]
    if DEBUG: 
        pass
        # print X[0], Y[0], Y[1]
        # print "Last:", X[len(X) - 1], Y[len(Y) - 1]
    return X, Y

def normalize_data(data):
    """"""
    # Create a deep copy to make sure original data is intact
    new_data = copy.deepcopy(data)
    try: 
        # will throw index error if data a vector
        cols = new_data.shape[1]
        for i in range(0, cols - 1): 
            mean = np.mean(new_data[:, i])
            sdt = np.std(new_data[:, i])
            # print mean, sdt
            new_data[:, i] = (new_data[:, i] - mean) / sdt
    except IndexError: 
        # Data is a vector. 
        mean = np.mean(new_data)
        sdt = np.std(new_data)
        new_data = (new_data - mean) / sdt
    return new_data

def mean_squared_loss(theta, x, y, regularised=False):
    """Compute mean squared loss"""
    data_count = np.shape(x)[0]
    hypothesis = np.array([np.dot(x_item, theta) for x_item in x])
    loss = hypothesis - y
    mean_squared_loss = np.sum(loss ** 2) / data_count   
    if regularised: 
        return LAMBDA * mean_squared_loss + (1 - LAMBDA) * np.sum(theta ** 2)
    else:        
        return mean_squared_loss

def squared_loss(theta, x, y, regularised=False):
    """Compute squared loss"""
    m, n = np.shape(x)
    hypothesis = np.array([np.dot(a, theta) for a in x])
    loss = hypothesis - y
    squared_loss = np.sum(loss ** 2)
    if regularised: 
        return LAMBDA * squared_loss + (1 - LAMBDA) * np.sum([th ** 2 for th in theta])
    else:        
        return squared_loss
    
def grad(theta, x, y):
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    gradient = np.dot(x.T, loss)
    return gradient

def feature_selection_financial_data(data):  
    """Feature selection"""
    m, n = np.shape(data)  
    slice_size = len(data[0]) / 2
    new_data = np.ones(shape=(len(data), 5))
    
    new_data[:, 0] = (data[: , slice_size ] - data[: , slice_size + 1])
    new_data[:, 1] = data[: , slice_size + 1] - data[: , slice_size + 2]
    new_data[:, 2] = data[: , 0 ] - data[: , 1 ]
    new_data[:, 3] = np.square(data[: , slice_size ])
    # for i in range(slice_size, n - 2): 
    #    new_data[:, i - slice_size + 1] = data[: , i] - data[: , i + 1]
            
    # new_data[:, 1] = data[: , slice_size] - data[: , slice_size + 1]
    # new_data[:, 2] = data[: , slice_size + 1] - data[: , slice_size + 2]
    # new_data[:, 3] = data[: , 0 ]
    
    return new_data

def slice_and_sum(data, col1, col2):
    data_slice = data[:, range(col1, col2)]
    return np.array([np.sum(row) / len(row) for row in data_slice])
    
def find_theta_linear(train_x, train_y, validate_x, validate_y, regularised=False, max_iterations=100):
    """"""
    features_count = np.shape(train_x)[1]
    theta = np.ones(features_count)
    theta = fmin_bfgs(mean_squared_loss, theta, args=(train_x, train_y, regularised), norm=float('-inf'), maxiter=max_iterations)
    print "Optimized theta:", theta
    train_msq = mean_squared_loss(theta, train_x, train_y, regularised)
    validate_msq = mean_squared_loss(theta, validate_x, validate_y, regularised)
    print "Train mean squared loss: ", train_msq
    print "Validate mean squared loss: ", validate_msq
    return theta, validate_msq
    
def linear(input_file_name):
    """Linear regression algorithm for Task 1."""
    print "Running linear regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name)
    x = feature_selection_financial_data(x)
    m, n = np.shape(x)    
    print "Shape of data: ", m, " rows, ", n, " cols."
    print x[0]
    # x = normalize_data(x)
    # y = normalize_data(y)
    
    fold_1_x = x[0 : len(x) / 2]
    fold_1_y = y[0 : len(y) / 2]
    fold_2_x = x[len(x) / 2 : len(x)]
    fold_2_y = y[len(y) / 2 : len(y)]
    # Get normalised data
    fold_1_x_n = normalize_data(x[0 : len(x) / 2])
    fold_1_y_n = normalize_data(y[0 : len(y) / 2])
    fold_2_x_n = normalize_data(x[len(x) / 2 : len(x)])
    fold_2_y_n = normalize_data(y[len(y) / 2 : len(y)])    
    # Cross-fold validation. 
    theta1, validate_msq1 = find_theta_linear(fold_1_x_n, fold_1_y_n, fold_2_x, fold_2_y)
    theta2, validate_msq2 = find_theta_linear(fold_2_x_n, fold_2_y_n, fold_1_x, fold_1_y)
    # Average mean squared error. 
    average_msl = (validate_msq1 + validate_msq2) / 2
    print "\n----\nAverage validate mean squared loss: ", average_msl
    
    if DEBUG: 
        x_print = np.array(np.arange(len(fold_2_y)))    
        line = np.zeros(len(fold_2_y))
        for i in range(n): 
            line += theta1[i] * fold_2_x_n[:, i]
        plt.plot(x_print, line, 'r-', x_print, fold_2_y_n, 'o')
        plt.show()
    return average_msl
        
def reglinear(input_file_name):
    x, y = read_data_file(input_file_name)
    x = feature_selection_financial_data(x)
    m, n = np.shape(x)    
    print "Shape of data: ", m, " rows, ", n, " cols."
    #x = normalize_data(x)
    #y = normalize_data(y)
    print x[0], y[0]
    
    fold_1_x = x[0 : len(x) / 2]
    fold_1_y = y[0 : len(y) / 2]
    fold_2_x = x[len(x) / 2 : len(x)]
    fold_2_y = y[len(y) / 2 : len(y)]
    # Get normalised data
    fold_1_x_n = normalize_data(fold_1_x)
    fold_1_y_n = normalize_data(fold_1_y)
    fold_2_x_n = normalize_data(fold_2_x)
    fold_2_y_n = normalize_data(fold_2_y)    
    # Cross-fold validation. 
    theta1, validate_msq1 = find_theta_linear(fold_1_x_n, fold_1_y_n, fold_2_x, fold_2_y, regularised=True)
    theta2, validate_msq2 = find_theta_linear(fold_2_x_n, fold_2_y_n, fold_1_x, fold_1_y, regularised=True)
    # Average mean squared loss. 
    average_msl = (validate_msq1 + validate_msq2) / 2
    print "\n----\nAverage validate mean squared loss: ", average_msl
    
    if DEBUG: 
        x_print = np.array(np.arange(len(fold_2_y)))    
        line = np.zeros(len(fold_2_y))
        for i in range(n): 
            line += theta1[i] * fold_2_x_n[:, i]
        plt.plot(x_print, line, 'r-', x_print, fold_2_y_n, 'o')
        plt.show()
        
    return average_msl
        
###################################################################################
# Logistic regression code
    
def class_probability(theta, x, class_ind):
    """Compute the probability for a class, given a data point x."""
    exp_vector = np.array([np.exp(np.dot(x, theta[j])) for j in range(theta.shape[0])])
    return  exp_vector[class_ind] / np.sum(exp_vector) 

def feature_selection_financial_data_logistic(data):  
    """Feature selection"""
    m, n = np.shape(data)  
    slice_size = len(data[0]) / 2
    new_data = np.ones(shape=(len(data), 5))
    
    # new_data[:, 0] = data[: , slice_size]
    new_data[:, 0] = (data[: , 0 ] - data[: , 1 ]) ** 3
    new_data[:, 1] = np.square(data[: , slice_size] - data[: , slice_size + 1]) 
    new_data[:, 2] = data[: , slice_size + 1] - data[: , slice_size + 2]
    new_data[:, 3] = (data[: , slice_size]) ** 11
    # new_data[:, 5] = np.square(data[: , 0])
    """
    new_data[:, 1] = data[: , slice_size + 1]
    new_data[:, 2] = data[: , slice_size] - data[: , slice_size + 1]
    new_data[:, 3] = data[: , slice_size + 1] - data[: , slice_size + 2]
    new_data[:, 4] = data[: , 0 ]
    new_data[:, 5] = data[: , 0 ] - data[: , 1 ]
    """
    # for i in range(slice_size, n - 2): 
    #    new_data[:, i - slice_size + 1] = data[: , i] - data[: , i + 1]
    
    print new_data[0]
        
    # new_data[:, 2] = data[: , slice_size + 1] - data[: , slice_size + 2]
    # new_data[:, 3] = data[: , 0 ]
    
    return new_data

def accuracy(theta, validate_x, validate_y):
    """Calculate accuracy of theta on a validate dataset. 
    validate_y contains the actual classes for comparison."""
    data_count = np.shape(validate_x)[0]
    class_count = theta.shape[0]  # Number of classes
    success = 0
    classes_count = {}  # Maintain class occurrences in a dictionary
    computed_y = []
    for i in range(0, data_count): 
        try: 
            classes_count[validate_y[i]] = classes_count[validate_y[i]] + 1
        except KeyError:  # no such key yet
            classes_count[validate_y[i]] = 1
        probabilities = [class_probability(theta, validate_x[i], j) for j in range(class_count)]
        # print probabilities
        winner = np.argmax(probabilities)  # Hard accuracy
        computed_y.append(winner)
        if int(validate_y[i]) == winner: 
            success += 1
    print "\nDataset representation by classes: "
    for cl in classes_count.keys(): 
        print "Class ", int(cl) , ": ", (float(classes_count[cl]) * 100.) / float(data_count), "%"         
    accuracy = (float(success) * 100.) / float(data_count)
    print "\nAccuracy: ", accuracy, "%"
    print computed_y
    return accuracy, computed_y

def find_theta_logistic(train_x, train_y, validate_x, validate_y, cost_function, regularise=False, max_iterations=20, no_classes=5):
    """Find optimal theta for a train dataset, and use the result to compute 
    the accuracy on a validate dataset."""
    features_count = np.shape(train_x)[1]
    theta = np.ones((no_classes, features_count))
    print "Initialized empty theta with shape:", theta.shape
    # Run the L-BFGS-B algorithm to find optimal values for theta. 
    # L-BFGS-B seems to perform better than BFGS in this case. 
    output = minimize(cost_function, theta, args=(train_x, train_y, regularise,), method='L-BFGS-B', jac=True, options={'maxiter': max_iterations, 'disp':False})
    theta = output.x.reshape((no_classes, features_count))
    print "Optimized Theta: ", theta
    validate_cost, grad = cost_function(theta, validate_x, validate_y)
    print "Cost for validate: ", validate_cost
    acc, y_computed = accuracy(theta, validate_x, validate_y)
    return theta, acc, validate_cost

def logistic(input_file_name):
    print "Running logistic regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name, True)  
    # x = feature_selection_financial_data_logistic(x)
    print "Data shape: ", y.shape, x.shape
    
    train_x = x[0 : len(x) / 2]
    train_y = y[0 : len(y) / 2]
    validate_x = x[len(x) / 2 : len(x)]
    validate_y = y[len(y) / 2 : len(y)]
    train_x_n = normalize_data(train_x)
    validate_x_n = normalize_data(validate_x)
    
    theta1, accuracy1, validate_cost_1 = find_theta_logistic(train_x_n, train_y, validate_x_n, validate_y, compute_cost, regularise=False)
    theta2, accuracy2, validate_cost_2 = find_theta_logistic(validate_x_n, validate_y, train_x_n, train_y, compute_cost, regularise=False)
    
    average_accuracy = (accuracy1 + accuracy2) / 2
    average_validate_cost = (validate_cost_1 + validate_cost_2) / 2
    print "\n-----\nAverage accuracy: ", average_accuracy , "%"
    print "Average validate cost: ", average_validate_cost , "%"
    return average_accuracy
    
def reglogistic(input_file_name):
    print "Running regularised logistic regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name, True)
    print "Data shape: ", y.shape, x.shape    
    x = feature_selection_financial_data_logistic(x)
    # 
    print x[0]
    train_x = x[0 : len(x) / 2]
    train_y = y[0 : len(y) / 2]
    validate_x = x[len(x) / 2 : len(x)]
    validate_y = y[len(y) / 2 : len(y)]
    train_x_n = normalize_data(train_x)
    validate_x_n = normalize_data(validate_x)    
    
    theta1, accuracy1, validate_cost_1 = find_theta_logistic(train_x_n, train_y, validate_x_n, validate_y, compute_cost, regularise=True)
    theta2, accuracy2, validate_cost_2 = find_theta_logistic(validate_x_n, validate_y, train_x_n, train_y, compute_cost, regularise=True)
        
    average_accuracy = (accuracy1 + accuracy2) / 2
    average_validate_cost = (validate_cost_1 + validate_cost_2) / 2
    print "\n-----\nAverage accuracy: ", average_accuracy , "%"
    print "Average validate cost: ", average_validate_cost , "%"
    return average_accuracy
    
def compute_class(current_price, previous_price):
    """Determine the class of a day given 
    the prices of current and previous days."""
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
        print "Current: ", current_price, " Previous: ", previous_price, " Chage: ", percentage, "Class: ", price_class
    return price_class
    
def identity(class1, class2):
    if class1 == class2: 
        return 1
    return 0

def compute_cost(theta, X, y, regularise=False, num_classes=5):
    """Cost function for logistic regression.
    Returns negated cost and gradient, because 
    is to be optimised by a minimisation algorithm."""
    global LAMBDA
    data_count, features_count = X.shape
    # Need to reshape, because optimisation algorithms flatten theta.  
    theta = theta.reshape((num_classes, features_count))
    loss = 0.
    grad = np.zeros(shape=theta.shape)
    for i in range(0, data_count):  # For every data point in X
        current_class = int(y[i])
        probability = class_probability(theta, X[i], current_class)
        loss += np.log(probability)           
        for k in range(0, num_classes):  # Update gradient for every class. 
            grad[k] += np.dot(X[i], (identity(current_class, k) - class_probability(theta, X[i], k)))                
    if regularise:  # Ridge regularisation
        loss = LAMBDA * loss - (1 - LAMBDA) * np.sum(np.absolute(theta) ** 2)
        grad = LAMBDA * grad - 2 * (1 - LAMBDA) * theta
    return [-loss, -grad.flatten()]
  
if __name__ == '__main__':
    global LAMBDA
    start_time = time.time()
    
    # linear('stock_price.csv')
    reglinear('stock_price.csv')
    # logistic('stock_price.csv')
    # reglogistic('stock_price.csv')
    
    elapsed_time = time.time() - start_time
    print "Total time:", elapsed_time, "seconds."
    
    
# EOF
