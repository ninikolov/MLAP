'''
Created on 15 Feb 2014

Code for MLAP Open Assessment, Part 1. 
Linear regression and Logistic regression. 

@author: Y6189686

'''

import copy
import csv
import time

from scipy.misc import logsumexp
from scipy.optimize import fmin_bfgs, minimize

import numpy as np

# Raise calculation errors as exceptions - such as floating point error. 
np.seterr(all='raise')
# Print precision. 
np.set_printoptions(threshold=np.nan, precision=4)

DEBUG = False
LAMBDA = 0.5

def read_data_file(input_file, calculate_class=False):
    """Read a csv data file. Used for both linear and logistic regression. 
    calculate_class determines whether to replace y values with a calculated class value."""
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        # Last column is x0 
        X = np.ones(shape=(len(rows) - 10, 21))
        if calculate_class: 
            Y = np.zeros(len(rows) - 10, dtype=np.uint8)
        else: 
            Y = np.zeros(len(rows) - 10)
        for i in range(0, len(rows)):
            if i < 10: 
                continue  # Skip first 10 rows
            # Push data to X and Y 
            for n in range(1, 11):  # 10 previous stock volumes
                X[i - 10][n - 1] = rows[i - n ][0]
            for n in range(11, 21):  # 10 previous prices
                X[i - 10][n - 1] = rows[i - (n - 10) ][1]
            if calculate_class:  # Calculate the class from the prices                
                Y[i - 10] = compute_class(float(rows[i][1]), float(rows[i - 1][1]))
            else:  # Use the price                
                Y[i - 10] = rows[i][1]
    return X, Y

def standardize_data(data):
    """Perform location and scale transform on each column of data."""
    # Create a deep copy to make sure original data is intact
    new_data = copy.deepcopy(data)
    try: 
        cols = new_data.shape[1]  # will throw index error if data a vector
        for i in range(0, cols - 1):  # For every column in new_data
            mean = np.mean(new_data[:, i])
            sdt = np.std(new_data[:, i])
            new_data[:, i] = (new_data[:, i] - mean) / sdt
    except IndexError:  # Data is a vector.         
        mean = np.mean(new_data)
        sdt = np.std(new_data)
        new_data = (new_data - mean) / sdt
    return new_data

def split_data_random(X, y):
    """Split data into two random folds. """
    rng = np.random.RandomState(np.random.randint(1, 500))
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)  # randomize indices
    # Split into two 
    fold_1_x = X[indices[0 : len(X) / 2]]
    fold_1_y = y[indices[0 : len(y) / 2]]
    fold_2_x = X[indices[len(X) / 2 : len(X)]]
    fold_2_y = y[indices[len(y) / 2 : len(y)]]
    return fold_1_x, fold_2_x, fold_1_y, fold_2_y

def mean_squared_loss(theta, x, y, regularised=False):
    """Compute mean squared loss"""
    data_count = np.shape(x)[0]
    hypothesis = np.array([np.dot(x_item, theta) for x_item in x])
    loss = hypothesis - y
    mean_squared_loss = np.sum(loss ** 2) / data_count
    if regularised:  # Ridge regularisation
        return LAMBDA * mean_squared_loss + (1 - LAMBDA) * np.sum(theta ** 2)
    else:        
        return mean_squared_loss

def squared_loss(theta, x, y, regularised=False):
    """Compute squared loss"""
    m, n = np.shape(x)
    hypothesis = np.array([np.dot(a, theta) for a in x])
    loss = hypothesis - y
    squared_loss = np.sum(loss ** 2)
    if regularised:  # Ridge regularisation
        return LAMBDA * squared_loss + (1 - LAMBDA) * np.sum([th ** 2 for th in theta])
    else:        
        return squared_loss

def feature_selection_financial_data(data):  
    """Feature selection for linear regression"""
    data_rows, data_cols = np.shape(data)
    data_middle = int(len(data[0]) / 2)    
    features_repeat = 0
    additional_features = 5
    total_features = data_cols * features_repeat - features_repeat + additional_features + 1   
        
    expanded_data = np.ones(shape=(data_rows, total_features))    
    expanded_data[:, 0] = data[: , data_middle ]
    expanded_data[:, 1] = data[: , data_middle ] - data[: , data_middle + 1]
    expanded_data[:, 2] = data[: , data_middle + 1] - data[: , data_middle + 2]
    expanded_data[:, 3] = data[:, 0] - data[:, 1]
    expanded_data[:, 4] = data[:, 1] - data[:, 2]
    """
    if features_repeat > 0: 
        repeat = data_middle * features_repeat + additional_features - 2
        for i in range(additional_features, repeat, features_repeat): 
            data_index = i / features_repeat
            expanded_data[:, i] = data[:, data_index]
            expanded_data[:, i + 1 ] = data[:, data_index] ** 2
            expanded_data[:, i + 2] = np.sqrt(data[:, data_index])
            if data_index < data_middle: 
                expanded_data[:, i + 3] = data[:, data_index] - data[:, data_index + 1]
        
        for i in range(repeat, repeat * 2 - additional_features, features_repeat): 
            data_index = i / features_repeat 
            expanded_data[:, i] = data[:, data_index]
            expanded_data[:, i + 1 ] = data[:, data_index] ** 2
            expanded_data[:, i + 2] = np.sqrt(data[:, data_index])
            if data_index < data_middle * 2:
                expanded_data[:, i + 3] = data[:, data_index] - data[:, data_index + 1]
    """
    
    """
    new_data = np.ones(shape=(len(data), 4))
    
    new_data[:, 0] = data[: , slice_size ] 
    new_data[:, 1] = (data[: , slice_size ] - data[: , slice_size + 1]) 
    new_data[:, 2] = data[: , slice_size + 1] - data[: , slice_size + 2]
    """
    """
    # new_data[:, 3] = np.log(data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) 
    # for i in range(slice_size, n - 2): 
    #    new_data[:, i - slice_size + 1] = data[: , i] - data[: , i + 1]
            
    # new_data[:, 1] = data[: , slice_size] - data[: , slice_size + 1]
    # new_data[:, 2] = data[: , slice_size + 1] - data[: , slice_size + 2]
    # new_data[:, 3] = data[: , 0 ]
    """
    
    return expanded_data

def slice_and_sum(data, col1, col2):
    data_slice = data[:, range(col1, col2)]
    return np.array([np.sum(row) / len(row) for row in data_slice])
    
def find_theta_linear(train_x, train_y, validate_x, validate_y, regularised=False, max_iterations=100):
    """Find optimal theta for a train dataset, then use result to compute 
    mean squared loss on a validate dataset.
    regularised - determine whether to regularise"""
    features_count = np.shape(train_x)[1]
    theta = np.ones(features_count)
    # Run the BFGS algorithm to find optimal values for theta. 
    theta = fmin_bfgs(mean_squared_loss, theta, args=(train_x, train_y, regularised), norm=float('-inf'), maxiter=max_iterations)
    if DEBUG: 
        print "Optimized theta:", theta
        
    train_msq = mean_squared_loss(theta, train_x, train_y, regularised)
    validate_msq = mean_squared_loss(theta, validate_x, validate_y, regularised)
    sql = squared_loss(theta, validate_x, validate_y, regularised)
    print "Train mean squared loss: ", train_msq
    print "Validate mean squared loss: ", validate_msq
    print "sqloss:", sql
    
    return theta, validate_msq
    
def linear(input_file_name):
    """Linear regression algorithm for Task 1."""
    print "Running linear regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name)
    x = feature_selection_financial_data(x)  # Feature selection
    print x[0]
    x = standardize_data(x) 
    y = y - np.mean(y) 
    
    m, n = np.shape(x)    
    print "Shape of data: ", m, " rows, ", n, " cols."
    
    fold_1_x, fold_2_x, fold_1_y, fold_2_y = split_data_random(x, y)
    
    # 2-fold cross-validation. 
    theta1, validate_msq1 = find_theta_linear(fold_1_x, fold_1_y, fold_2_x, fold_2_y)
    theta2, validate_msq2 = find_theta_linear(fold_2_x, fold_2_y, fold_1_x, fold_1_y)
    # Average mean squared error for the 2 folds. 
    average_msl = (validate_msq1 + validate_msq2) / 2
    print "\n----\nAverage validate mean squared loss: ", average_msl

    return average_msl
        
def reglinear(input_file_name):
    """Regularised linear regression algorithm for Task 3."""
    print "Running linear regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name)
    x = feature_selection_financial_data(x)
    # print x[0], y[0]
    m, n = np.shape(x) 
    print "Shape of data: ", m, " rows, ", n, " cols."
    x = standardize_data(x)
    y = y - np.mean(y)
    
    fold_1_x, fold_2_x, fold_1_y, fold_2_y = split_data_random(x, y)
    
    # 2-fold cross-validation. 
    theta1, validate_msq1 = find_theta_linear(fold_1_x, fold_1_y, fold_2_x, fold_2_y, regularised=True)
    theta2, validate_msq2 = find_theta_linear(fold_2_x, fold_2_y, fold_1_x, fold_1_y, regularised=True)
    # Average mean squared error for the 2 folds. 
    average_msl = (validate_msq1 + validate_msq2) / 2
    print "\n----\nAverage validate mean squared loss: ", average_msl
    
    return average_msl
        
###################################################################################
# Logistic regression code

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
        pass
        # print "Current: ", current_price, " Previous: ", previous_price, " Chage: ", percentage, "Class: ", price_class
    return price_class
    
def class_probability(theta, x, class_ind):
    """Compute the probability for a class, given a data point x."""
    try: 
        exp_vector = np.array([np.exp(np.dot(x, theta[j])) for j in range(theta.shape[0])])
        return  exp_vector[class_ind] / np.sum(exp_vector) 
    except FloatingPointError or OverflowError:  # Overflow for exp
        return class_probability_failsafe(theta, x, class_ind)

def class_probability_failsafe(theta, x, class_ind):
    """Alternative probability calculation method with logsumexp. 
    Shouldn't normally get to here."""
    dot_vector = np.array([np.dot(theta[j], x) for j in range(0, theta.shape[0])])
    return np.exp(dot_vector[class_ind] - logsumexp(dot_vector))
        
def feature_selection_financial_data_logistic(data):  
    """Feature selection"""
    data_rows, data_cols = np.shape(data)
    data_middle = int(len(data[0]) / 2)    
    features_repeat = 0
    additional_features = 1
    total_features = data_cols * features_repeat / 2 - features_repeat / 2 + additional_features + 1   
        
    expanded_data = np.ones(shape=(data_rows, total_features))    
    expanded_data[:, 0] = data[: , data_middle ] - data[: , data_middle + 1]
    
    if features_repeat > 0: 
        repeat = data_middle * features_repeat + additional_features
        for i in range(additional_features, repeat, features_repeat): 
            data_index = i / features_repeat
            expanded_data[:, i] = data[:, data_index]
            expanded_data[:, i + 1] = data[:, data_index] ** 2
            expanded_data[:, i + 2] = np.sqrt(data[:, data_index])
            expanded_data[:, i + 3] = data[:, data_index] - data[:, data_index + 1]
        """
        for i in range(repeat, repeat * 2 - additional_features, features_repeat): 
            data_index = i / features_repeat - 1
            expanded_data[:, i] = data[:, data_index]
            expanded_data[:, i + 1] = data[:, i / data_index] ** 2
            expanded_data[:, i + 2] = data[:, i / data_index] ** 3
            expanded_data[:, i + 3] = data[:, i / data_index] - data[:, data_index + 1]
        """
    """
    expanded_data = np.ones(shape=(len(data), data_cols * 4 - 7))    
    for i in range(0, data_cols * 4 - 8, 4): 
        expanded_data[:, i] = data[:, i / 4]
        expanded_data[:, i + 1] = data[:, i / 4] ** 2
        expanded_data[:, i + 2] = data[:, i / 4] ** 3
        expanded_data[:, i + 3] = data[:, i / 4] - data[:, (i / 4) + 1]
        """
    """
    expanded_data = np.ones(shape=(len(data), 7))    
    # expanded_data[:, 0] = data[: , data_middle]
    expanded_data[:, 0] = data[: , data_middle ]
    expanded_data[:, 1] = (data[: , data_middle ] - data[: , data_middle + 1]) 
    expanded_data[:, 2] = data[: , data_middle + 1] - data[: , data_middle + 2]
    expanded_data[:, 3] = data[:, 1] - data[:, 0]
    expanded_data[:, 4] = data[:, 2] - data[:, 1]
    expanded_data[:, 5] = data[:, 3] - data[:, 2]
    # expanded_data[:, 6] = data[: , data_middle + 1]
    # expanded_data[:, 3] = (data[: , 0 ] - data[: , 1 ]) 
    # expanded_data[:, 5] = np.square(data[: , 0])
    """
    """
    expanded_data[:, 1] = data[: , data_middle + 1]
    expanded_data[:, 2] = data[: , data_middle] - data[: , data_middle + 1]
    expanded_data[:, 3] = data[: , data_middle + 1] - data[: , data_middle + 2]
    expanded_data[:, 4] = data[: , 0 ]
    expanded_data[:, 5] = data[: , 0 ] - data[: , 1 ]
    """
    
    return expanded_data

def logistic_accuracy(theta, validate_x, validate_y):
    """Calculate accuracy of theta on a validate dataset. 
    validate_y contains the actual classes for comparison."""
    data_count = np.shape(validate_x)[0]
    class_count = theta.shape[0]  # Number of classes
    successful_estimates = 0  # Count number of successful estimates of a class for theta
    classes_count = {}  # Maintain class occurrences in a dictionary
    computed_y = []
    for i in range(0, data_count): 
        try: 
            classes_count[validate_y[i]] = classes_count[validate_y[i]] + 1
        except KeyError:  # no such key yet
            classes_count[validate_y[i]] = 1
        probabilities = [class_probability(theta, validate_x[i], j) for j in range(class_count)]
        winner = np.argmax(probabilities)  # Hard accuracy
        computed_y.append(winner)
        if validate_y[i] == winner:  # If actual class matches our estimate
            successful_estimates += 1
    accuracy = float(successful_estimates) / float(data_count)    
    # print computed_y
    return accuracy, computed_y, classes_count

def print_classes_count(classes_count, data_count, dataset_name=""):
    """A helper method to print information about the classes contained 
    in a dataset."""
    print dataset_name, "dataset representation by classes: "
    for cl in classes_count.keys(): 
        print "Class ", int(cl) , ": ", (float(classes_count[cl])) / float(data_count)

def find_theta_logistic(train_x, train_y, validate_x, validate_y, cost_function, regularise=False, max_iterations=50, no_classes=5, display=DEBUG):
    """Find optimal theta for a train dataset, and use the result to compute 
    the accuracy on a validate dataset."""
    data_count, features_count = np.shape(train_x)
    theta = np.ones((no_classes, features_count))
    print "Initialized empty theta with shape:", theta.shape
    print "Running L-BFGS-B..."
    # Run the L-BFGS-B algorithm to find optimal values for theta. 
    # L-BFGS-B seems to perform better than BFGS in this case. 
    output = minimize(cost_function, theta, args=(train_x, train_y, regularise,), method='L-BFGS-B', jac=True, options={'maxiter':max_iterations, 'disp':display})
    theta = output.x.reshape((no_classes, features_count))
    if DEBUG: 
        print "Optimized Theta: ", theta    
    # Calculate accuracy and print relevant information
    train_accuracy, train_y_computed, train_classes = logistic_accuracy(theta, train_x, train_y)
    print "\n----\nTrain Accuracy: ", train_accuracy, ""
    validate_cost, grad = cost_function(theta, validate_x, validate_y)
    print "\nLogistic cost for validate: ", validate_cost
    validate_accuracy, y_computed, validate_classes = logistic_accuracy(theta, validate_x, validate_y)
    print_classes_count(validate_classes, data_count, "Validate")
    print "\nValidate Accuracy: ", validate_accuracy, "\n----"

    return theta, validate_accuracy, validate_cost

def logistic(input_file_name):
    print "Running logistic regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name, True)  
    x = feature_selection_financial_data(x)
    print x[0]
    x = standardize_data(x)
    print "Data shape: ", y.shape, x.shape
    
    fold_1_x, fold_2_x, fold_1_y, fold_2_y = split_data_random(x, y)

    theta1, accuracy1, validate_cost_1 = find_theta_logistic(fold_1_x, fold_1_y, fold_2_x, fold_2_y, compute_cost_logistic, regularise=False)
    theta2, accuracy2, validate_cost_2 = find_theta_logistic(fold_2_x, fold_2_y, fold_1_x, fold_1_y, compute_cost_logistic, regularise=False)
    
    average_accuracy = (accuracy1 + accuracy2) / 2
    average_validate_cost = (validate_cost_1 + validate_cost_2) / 2
    print "\n-----\nAverage accuracy: ", average_accuracy , "%"
    print "Average validate cost: ", average_validate_cost , "%"
    return average_accuracy
    
def reglogistic(input_file_name):
    print "Running regularised logistic regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name, True)
    x = feature_selection_financial_data(x)
    print x[0]
    print "Features shape: ", x.shape    
    # 
    x = standardize_data(x)
    
    fold_1_x, fold_2_x, fold_1_y, fold_2_y = split_data_random(x, y)
    
    theta1, accuracy1, validate_cost_1 = find_theta_logistic(fold_1_x, fold_1_y, fold_2_x, fold_2_y, compute_cost_logistic, regularise=True)
    theta2, accuracy2, validate_cost_2 = find_theta_logistic(fold_2_x, fold_2_y, fold_1_x, fold_1_y, compute_cost_logistic, regularise=True)
        
    average_accuracy = (accuracy1 + accuracy2) / 2
    average_validate_cost = (validate_cost_1 + validate_cost_2) / 2
    print "----\nAverage accuracy for cross-validation: ", average_accuracy , ""
    print "Average validate cost for cross-validation: ", average_validate_cost 
    return average_accuracy
    
def identity(class1, class2):
    if class1 == class2: 
        return 1
    return 0

def compute_cost_logistic(theta, X, y, regularise=False, num_classes=5):
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
        current_class = y[i]
        probability = class_probability(theta, X[i], current_class)
        loss += np.log(probability) 
        for k in range(0, num_classes):  # Update gradient for every class. 
            grad[k] += np.dot(X[i], (identity(current_class, k) - class_probability(theta, X[i], k)))                
    if regularise:  # Ridge regularisation
        loss = LAMBDA * loss - (1 - LAMBDA) * np.sum(np.absolute(theta) ** 2)
        grad = LAMBDA * grad - 2 * (1 - LAMBDA) * theta
    return [-loss, -grad.flatten()]

def test_all():
    global LAMBDA
    linear_acc = linear('stock_price.csv')
    logistic_acc = logistic('stock_price.csv')
    reg_acc_lin = {}
    reg_acc_log = {}
    for l in [0.3, 0.5, 0.8]: 
        LAMBDA = l
        acc2 = reglinear('stock_price.csv')
        reg_acc_lin[l] = acc2  
        acc2 = reglogistic('stock_price.csv')
        reg_acc_log[l] = acc2   
    
    print "Linear: ", linear_acc
    for l, acc in reg_acc_lin.items(): 
        print "Reglin with ", l, ":", acc
    print "Logistic: ", logistic_acc
    for l, acc in reg_acc_log.items(): 
        print "Reglog with ", l, ":", acc

if __name__ == '__main__':
    start_time = time.time()

    #linear_acc = linear('stock_price.csv')
    #reglinear_acc = reglinear('stock_price.csv')
    #logistic_acc = logistic('stock_price.csv')
    #acc2 = reglogistic('stock_price.csv')
    
    test_all()
    
    elapsed_time = time.time() - start_time
    print "Total execution time:", elapsed_time, "seconds."
    
    
# EOF
