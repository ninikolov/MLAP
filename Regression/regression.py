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
from scipy.optimize import minimize

import numpy as np

# Raise calculation errors as exceptions - such as floating point error. 
np.seterr(all='raise')
# Print precision. 
np.set_printoptions(threshold=np.nan, precision=4)

# Determines whether extra debug information should be printed about the execution
# of algorithms and minimisation routines. 
DEBUG = True
# Regularisation parameter
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
            Y = np.zeros(len(rows) - 10, dtype=np.uint8)  # classes are forced to be int
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

def standardise_data(data):
    """Perform location and scale transform on each column of data
    by subtracting the mean and dividing by the sdt deviation of 
    each column."""
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
    # return X[0 : len(X) / 2], X[len(X) / 2 : len(X)], y[0 : len(y) / 2], y[len(y) / 2 : len(y)]
    rng = np.random.RandomState(np.random.randint(1, 500))
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)  # randomise indices
    # Split into two according to randomised indices
    fold_1_x = X[indices[0 : len(X) / 2]]
    fold_1_y = y[indices[0 : len(y) / 2]]
    fold_2_x = X[indices[len(X) / 2 : len(X)]]
    fold_2_y = y[indices[len(y) / 2 : len(y)]]
    return fold_1_x, fold_2_x, fold_1_y, fold_2_y

def mean_squared_loss(theta, x, y, regularised=False):
    """Compute mean squared loss"""
    global LAMBDA
    data_count = np.shape(x)[0]
    hypothesis = np.dot(x, theta)
    loss = y - hypothesis
    mean_squared_loss = np.sum(np.square(loss)) / data_count
    if regularised:  # Ridge regularisation
        return LAMBDA * mean_squared_loss + (1 - LAMBDA) * np.sum(np.square(theta))
    else: 
        return mean_squared_loss

def squared_loss(theta, x, y, regularised=False):
    """Compute squared loss"""
    global LAMBDA
    hypothesis = np.dot(x, theta)
    loss = y - hypothesis
    squared_loss = np.sum(np.square(loss))
    if regularised:  # Ridge regularisation
        return LAMBDA * squared_loss + (1 - LAMBDA) * np.sum(np.square(theta))
    else:        
        return squared_loss

def feature_selection_financial_data(data):  
    """Feature selection and expansion. 
    This method works manually, by first creating a new matrix of desired size to contain
    the new features, and by then adding all the features. Last feature is x0 and should 
    be ones only. """
    data_rows, data_cols = np.shape(data)
    data_middle = int(len(data[0]) / 2)  # Represents the starting index of the stock price
    features_repeat = 0  # Number of features that will be looped. 
    additional_features = 3  # Number of additional features that are added manually
    # Calculate total number of features
    total_features = data_cols * features_repeat - features_repeat + additional_features + 1
    if features_repeat > 0: 
        total_features -= 2
    # Create output matrix to be filled with the features
    expanded_data = np.ones(shape=(data_rows, total_features))    
    
    # Feature set 9
    expanded_data[:, 0] = data[: , data_middle ]  # stock price for previous day
    expanded_data[:, 1] = data[: , data_middle ] - data[: , data_middle + 1]
    expanded_data[:, 2] = data[: , data_middle + 1] - data[: , data_middle + 2]
    # End feature set 9. 
#     expanded_data[:, 3] = data[:, 0] - data[:, 1] 
#     expanded_data[:, 4] = data[:, 1] - data[:, 2]
#     expanded_data[:, 5] = data[:, 0] 
#     expanded_data[:, 6] = np.sqrt(data[: , data_middle + 1 ])
#     expanded_data[:, 7] = data[: , data_middle + 2 ]
#     expanded_data[:, 8] = data[:, 1]
#     expanded_data[:, 9] = slice_and_sum(data, data_middle, data_middle * 2)
    
    if features_repeat > 0: 
        repeat = data_middle * features_repeat + additional_features - 2
        # Feature set 4
        # Loop through and add features for each of the original columns
        for i in range(additional_features, repeat, features_repeat): 
            data_index = i / features_repeat
            expanded_data[:, i] = data[:, data_index] 
            expanded_data[:, i + 1 ] = data[:, data_index + 1] ** 2
            expanded_data[:, i + 2] = np.sqrt(data[:, data_index])
            if data_index < data_middle: 
                expanded_data[:, i + 3] = data[:, data_index] - data[:, data_index + 1]
           
        for i in range(repeat, repeat * 2 - additional_features, features_repeat): 
            data_index = i / features_repeat 
            expanded_data[:, i] = data[:, data_index] 
            expanded_data[:, i + 1 ] = data[:, data_index + 1] ** 2
            expanded_data[:, i + 2] = np.sqrt(data[:, data_index])
            if data_index < data_middle * 2:
                expanded_data[:, i + 3] = data[:, data_index] - data[:, data_index + 1]    
    return expanded_data

def slice_and_sum(data, col1, col2):
    """Create a vector of averaged sums for each row between col1 and col2 in data."""
    data_slice = data[:, range(col1, col2)]
    return np.array([np.sum(row) / len(row) for row in data_slice])
    
def find_theta_linear(train_x, train_y, validate_x, validate_y, regularised=False, max_iterations=100):
    """Find optimal theta for a train dataset, then use result to compute 
    mean squared loss on a validate dataset.
    regularised - determine whether to regularise"""
    features_count = train_x.shape[1]
    theta = np.ones(features_count)
    # Run the L-BFGS-B algorithm to find optimal values for theta. 
    output = minimize(squared_loss, theta, args=(train_x, train_y, regularised,), method='L-BFGS-B', options={'maxiter':max_iterations, 'disp':DEBUG})
    theta = output.x
    if DEBUG: 
        print "Optimized theta:", theta        
    validate_msq = mean_squared_loss(theta, validate_x, validate_y)
    print "Validate mean squared loss: ", validate_msq    
    return theta, validate_msq
    
def linear(input_file_name):
    """Linear regression algorithm for Task 1."""
    print "Running linear regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name)
    x = feature_selection_financial_data(x)  # Feature selection
    x = standardise_data(x)  # Standardise data
    y = y - np.mean(y)  # Mean center y
    if DEBUG: 
        print x[0]
    m, n = np.shape(x)    
    print "Shape of data: ", m, " rows, ", n, " cols."    
    # Split data into two random folds
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
    m, n = np.shape(x)
    print "Shape of data: ", m, " rows, ", n, " cols."
    x = standardise_data(x)
    y = y - np.mean(y)  
    if DEBUG: 
        print x[0]  
    # Split data into two random folds
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
    except FloatingPointError or OverflowError:  # Overflow for exp. Try alternative method.
        return class_probability_failsafe(theta, x, class_ind)

def class_probability_failsafe(theta, x, class_ind):
    """Alternative probability calculation method with logsumexp. 
    Shouldn't normally get to here."""
    dot_vector = np.array([np.dot(theta[j], x) for j in range(0, theta.shape[0])])
    return np.exp(dot_vector[class_ind] - logsumexp(dot_vector))

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
        # Calculate probabilities for current data point
        probabilities = [class_probability(theta, validate_x[i], j) for j in range(class_count)]
        winner = np.argmax(probabilities)  # Hard accuracy
        computed_y.append(winner)
        if validate_y[i] == winner:  # If actual class matches our estimate
            successful_estimates += 1
    accuracy = float(successful_estimates) / float(data_count)    
    return accuracy, computed_y, classes_count

def print_classes_count(classes_count, data_count, dataset_name=""):
    """A helper method to print information about the classes contained 
    in a dataset."""
    print dataset_name, "dataset representation by classes: "
    for cl in classes_count.keys(): 
        print "Class ", int(cl) , ": ", (float(classes_count[cl])) / float(data_count)

def find_theta_logistic(train_x, train_y, validate_x, validate_y, regularise=False, max_iterations=200, no_classes=5, display=DEBUG):
    """Find optimal theta for a train dataset, and use the result to compute 
    the accuracy on a validate dataset."""
    data_count, features_count = np.shape(train_x)
    theta = np.ones((no_classes, features_count))
    print "Initialized empty theta with shape:", theta.shape
    print "Running L-BFGS-B..."
    # Run the L-BFGS-B algorithm to find optimal values for theta. 
    output = minimize(compute_cost_logistic, theta, args=(train_x, train_y, regularise,), method='L-BFGS-B', jac=True, options={'maxiter':max_iterations, 'disp':display})
    theta = output.x.reshape((no_classes, features_count))
    if DEBUG: 
        print "Optimized Theta: ", theta    
    # Calculate accuracy and print relevant information
    train_accuracy, train_y_computed, train_classes = logistic_accuracy(theta, train_x, train_y)
    print "\n----\nTrain Accuracy: ", train_accuracy, ""
    validate_cost, grad = compute_cost_logistic(theta, validate_x, validate_y)
    print "\nLogistic cost for validate: ", validate_cost
    validate_accuracy, y_computed, validate_classes = logistic_accuracy(theta, validate_x, validate_y)
    print_classes_count(validate_classes, data_count, "Validate")
    if DEBUG: 
        print "Computed y:", y_computed    
    print "\nValidate Accuracy: ", validate_accuracy, "\n----"
    return theta, validate_accuracy, validate_cost

def logistic(input_file_name):
    """Logistic regression algorithm for Task 2"""
    print "Running logistic regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name, True)  
    x = feature_selection_financial_data(x)
    x = standardise_data(x)  # Standardise data
    if DEBUG: 
        print x[0]
    print "Data shape: ", y.shape, x.shape
    # Split data into two random folds
    fold_1_x, fold_2_x, fold_1_y, fold_2_y = split_data_random(x, y)
    # Cross-validation
    theta1, accuracy1, validate_cost_1 = find_theta_logistic(fold_1_x, fold_1_y, fold_2_x, fold_2_y, regularise=False)
    theta2, accuracy2, validate_cost_2 = find_theta_logistic(fold_2_x, fold_2_y, fold_1_x, fold_1_y, regularise=False)
    # Calculate average accuracy across the 2 validate folds
    average_accuracy = (accuracy1 + accuracy2) / 2
    average_validate_cost = (validate_cost_1 + validate_cost_2) / 2
    print "\n-----\nAverage accuracy: ", average_accuracy , "%"
    print "Average validate cost: ", average_validate_cost , "%"
    return average_accuracy
    
def reglogistic(input_file_name):
    """Regularised logistic regression algorithm for Task 3"""
    print "Running regularised logistic regression on", input_file_name, ".\n"
    x, y = read_data_file(input_file_name, True)
    x = feature_selection_financial_data(x)
    print x[0]
    print "Features shape: ", x.shape        
    x = standardise_data(x)  # Standardise data. 
    if DEBUG: 
        print x[0]
    # Split data into two random folds
    fold_1_x, fold_2_x, fold_1_y, fold_2_y = split_data_random(x, y)
    # Cross-validation
    theta1, accuracy1, validate_cost_1 = find_theta_logistic(fold_1_x, fold_1_y, fold_2_x, fold_2_y, regularise=True)
    theta2, accuracy2, validate_cost_2 = find_theta_logistic(fold_2_x, fold_2_y, fold_1_x, fold_1_y, regularise=True)
    # Calculate average accuracy across the 2 validate folds
    average_accuracy = (accuracy1 + accuracy2) / 2
    average_validate_cost = (validate_cost_1 + validate_cost_2) / 2
    print "----\nAverage accuracy for cross-validation: ", average_accuracy , ""
    print "Average validate cost for cross-validation: ", average_validate_cost 
    return average_accuracy
    
def identity(class1, class2):
    if class1 == class2: 
        return 1
    return 0

def compute_cost_logistic_safe(theta, X, y, regularise=False, num_classes=5):
    """Cost function for logistic regression.
    Returns negated cost and gradient, because 
    is to be optimised by a minimisation algorithm.
    This one should be safer to compute with, but slower than compute_cost_logistic."""
    global LAMBDA
    data_count, features_count = X.shape
    # Need to reshape, because optimisation algorithms flatten theta.  
    theta = theta.reshape((num_classes, features_count))
    loss = 0.
    grad = np.zeros(shape=theta.shape)
    for i in range(0, data_count):  # For every data point in X
        current_class = y[i]
        probabilities = [class_probability(theta, X[i], j) for j in range(num_classes)]
        loss += np.log(probabilities[current_class]) 
        for k in range(0, num_classes):  # Update gradient for every class. 
            grad[k] += np.dot(X[i], (identity(current_class, k) - probabilities[k]))                
    if regularise:  # Ridge regularisation
        loss = LAMBDA * loss - (1 - LAMBDA) * np.sum(np.square(np.absolute(theta)))
        grad = LAMBDA * grad - 2 * (1 - LAMBDA) * theta
    return [-loss, -grad.flatten()]

def compute_cost_logistic(theta, X, y, regularise=False, num_classes=5):
    """Cost function for logistic regression.
    Returns negated cost and gradient, because 
    is to be optimised by a minimisation algorithm.
    Faster than compute_cost_logistic_safe."""
    global LAMBDA
    data_count, features_count = X.shape
    # Need to reshape, because optimisation algorithms flatten theta.  
    theta = theta.reshape((num_classes, features_count))
    grad = np.zeros(shape=theta.shape)
    dot_theta_x = np.dot(theta, X.T)
    try: 
        hypothesis = np.exp(dot_theta_x)
    except FloatingPointError: 
        print "FloatingPointError, using compute_cost_logistic_safe function instead"
        return compute_cost_logistic_safe(theta, X, y, regularise, num_classes)
    # Compute probabilities matrix. 
    probabilities = np.transpose(hypothesis / np.sum(hypothesis, axis=0))
    # Vector of actual probabilities
    actual_prob = np.array([probabilities[i][y[i]] for i in range(data_count)])
    loss = np.sum(np.log(actual_prob))  # Logistic loss    
    for i in range(0, data_count):
        for k in range(0, num_classes):  # Update gradient for every class. 
            grad[k] += np.dot(X[i], (identity(y[i], k) - probabilities[i][k])) 
    if regularise:  # Ridge regularisation
        loss = LAMBDA * loss - (1 - LAMBDA) * np.sum(np.square(np.absolute(theta)))
        grad = LAMBDA * grad - 2 * (1 - LAMBDA) * theta
    return [-loss, -grad.flatten()]
 
def lambda_test(precision=0.05, average_over=1, target_function=reglinear):
    """A helper function to test models with different lambdas.
    Executes average_over iterations, with lambda ranging from 0 to 1, with precision 
    specified by the precision parameter.
    target_function - the target function, can be reglinear or reglogistic."""
    global LAMBDA
    lambdas = np.arange(0, 1.0 + precision, precision)  # generate array of lambdas to test on
    averaged_array = np.zeros(shape=(average_over, len(lambdas)))
    for i in range(average_over): 
        errors = []
        for l in lambdas: 
            LAMBDA = l
            print "LAMBDA=", l
            # Run target function and store result
            errors.append(target_function('stock_price.csv'))
        averaged_array[i] = errors
    avg_errors = np.average(averaged_array, axis=0)
    l = 0.
    print "----\nResults for lambda:\n"
    for error in avg_errors:  # print each averaged error
        print "Reglin with ", l, ":", error
        l += precision
    print "----"
        
if __name__ == '__main__':
    start_time = time.time()

    linear('stock_price.csv')
    # reglinear('stock_price.csv')
    #logistic('stock_price.csv')
    #reglogistic('stock_price.csv')
    
    #lambda_test(average_over=5)
    
    elapsed_time = time.time() - start_time
    print "Total execution time:", elapsed_time, "seconds."
    
    
# EOF
