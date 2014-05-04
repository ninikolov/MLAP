'''
Created on 25 Feb 2014

Code for MLAP Open Assessment, Part 2. 
Bayesian networks. 

@author: Y6189686
'''

from collections import OrderedDict
import csv
import itertools
import random
import time

import numpy as np


def read_data_file(input_file):
    """Read a csv data file and produce a numpy ndarray. 
    0s and 1s are expected. 
    """
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        # Generate empty ndarray, forcing the type to be int
        data = np.zeros((len(rows), len(rows[0])), dtype=np.int)
        for i in range(0, len(rows)):
            data[i] = rows[i]    
    return data
    
def read_conditions(structure, index):
    """Compose a list of conditions for a variable, 
    from a given structure of a BN."""
    conditions = []
    for item in range(structure.shape[1]): 
        if structure[item][index] == 1:
            conditions.append(item)
    return conditions

def print_probability(conditions, probability, var_index):
    """Format variable and probability for display."""
    if not conditions:  # If variable has no conditions
        print "Probability(", var_index, " = 0 ) =", 1 - probability
        print "Probability(", var_index, " = 1 ) =", probability
        return
    condition_str = ""
    for condition, bool_value in conditions: 
        condition_str = condition_str + str(condition) + " = " + str(bool_value) + "; "
    condition_str = condition_str[:-2]
    print "Probability(", var_index, " = ", 1, "|", condition_str, ") =", probability
    print "Probability(", var_index, " = ", 0, "|", condition_str, ") =", 1 - probability

def calculate_conditional_prob(var_index, cond_list, data, alpha=1., beta=1.):
    """Calculate conditional probabilities for a variable with a Bayesian approach.     
    """
    rows = data.shape[0]
    # Output is an ordered dictionary, containing probability for every
    # condition combination. OrderedDict can be sorted, which is useful. 
    output = OrderedDict()
    # Generate all combinations for values of conditions
    var_combinations = list(itertools.product([0, 1], repeat=len(cond_list)))
    for combination in var_combinations: 
        # Keep track of 0 and 1 occurrences, relevant to the conditional 
        # probability that's examined. 
        count_1_occurances = 0
        count_0_occurances = 0
        # Possible values for combinations. 
        # Those are tuples with the variable index as first value, and the boolean value as the second.
        values = zip(cond_list, combination)  
        for row in range(rows): 
            success = True
            # Loop over conditions and their boolean value
            for condition, bool_value in values: 
                # If value not as expected, break
                if data[row][condition] != bool_value: 
                    success = False
                    break
            if success:  # If we've found value that should be counted update our counts. 
                if data[row][var_index] == 1: 
                    count_1_occurances += 1
                else: 
                    count_0_occurances += 1
        # Calculate probability from counts
        probability = (alpha + float(count_1_occurances)) / (alpha + beta + float(count_1_occurances + count_0_occurances))
        output[tuple(values)] = probability
        print_probability(values, probability, var_index)
    return output
    
def estimate_parameter(structure, data, index, alpha=1., beta=1.):    
    """Estimate parameter at index. 
    Uses a bayesian approach. Implemented as described in section
    9.4 of BRML book (page 199). 
    Uses an uniform prior - alpha and beta are 1. 
    """
    rows = data.shape[0]
    conditions = read_conditions(structure, index)
    if not conditions:  # if target variable has no conditions calculate directly
        total = np.sum(data[:, index])  # count 1s at index
        prob = (alpha + float(total)) / (alpha + beta + float(rows))
        print_probability([], prob, index)        
        return [prob]
    else: 
        return calculate_conditional_prob(index, conditions, data, alpha, beta)

def sample(prob):
    """Draw a random number and compare it to prob. """
    rand = random.random()
    if rand < prob: 
        return 1
    return 0

def conditional_sample(var_index, probability_dict, current_dict):
    """Sample conditional probability.
    var_index - index of variable we're looking at
    probability_dict - OrderedDict containing the estimated probabilities of the variable
    current_dict - dictionary of already estimated values for variables"""
    output = current_dict
    remaining = []  # list monitoring the variables, which haven't been sampled yet
    for condition_tuples, probability in probability_dict.items(): 
        found = True
        for condition in condition_tuples: 
            if condition[0] in output:  # If we've already estimated the value for this variable
                # If the estimated value isn't for the current condition's value
                if output[condition[0]] != condition[1]:  
                    found = False
                    break
            else:  # If condition hasn't been estimated yet
                if not var_index in remaining:  # If not in remaining
                    remaining.append(var_index)
                found = False
                break
        if found:  # If we've found the right condition, and it has a value
            output[var_index] = sample(probability)  # Sample with its probability and add to output
    return output, remaining
            
def ancestral_sampling(network, predefined=None):
    """Implements ancestral sampling as described in slide 8 of the MLAP 
    Markov Chain Monte Carlo lecture.
    predefined variable is used to specify already sampled values of the BN to use"""
    # Sort by length
    network = dict(sorted(network.iteritems(), key=lambda x: len(x[1])))
    if predefined: 
        output = predefined
    else: 
        output = {}
    remaining = []
    for key, item in network.items():  # for each pair of variable and probability representation of the BN
        if isinstance(item, list):  # not a conditional probability
            output[key] = sample(item[0])
        elif isinstance(item, OrderedDict):  # conditional probability
            output, missing_conditions = conditional_sample(key, item, output)
            remaining = remaining + missing_conditions
    if remaining:  # Any values not determined? Go over network again. 
        # Use recursion to estimate values for any remaining variables. 
        output = dict(output.items() + ancestral_sampling(network, output).items())
    return output
    
def bnbayesfit(structure_file_name, data_file_name):
    """Estimate parameters of bayesian network, defined in structure_file_name
    by using data from data_file_name.
    data_file_name can also be a numpy ndarray."""
    # Structure of BN
    structure = read_data_file(structure_file_name)
    if not isinstance(data_file_name, np.ndarray): 
        data = read_data_file(data_file_name)
    else:  # Data can also be an ndarray, rather than a file.  
        data = data_file_name
    rows, cols = data.shape
    print "Data is", rows, "rows", cols, "cols."
    # fittedbn is dictionary of structures
    fittedbn = {} 
    for i in range(cols): 
        fittedbn[i] = estimate_parameter(structure, data, i)
    print fittedbn
    return fittedbn
    
def bnsample(fittedbn, nsamples):
    """Generates samples for a fitted bayesian network fittedbn. 
    Uses ancestral sampling. 
    nsamples - number of samples to generate"""
    output = np.empty((nsamples, len(fittedbn)), dtype=np.int)  # generate empty matrix
    for i in range(nsamples): 
        sample_dict = ancestral_sampling(fittedbn) 
        # Replace row with generated sample
        output[i] = np.array([sample_dict[key] for key in sorted(sample_dict.keys())])
    print output
    return output
    
def learn_from_samples(fittedbn, nsamples, structure_file_name):
    """Use samples to learn parameters of a bayesian network. 
    Do a fit from the generated samples. """
    samples = bnsample(fittedbn, nsamples)
    second_fit = bnbayesfit(structure_file_name, samples)
    return second_fit

if __name__ == '__main__':
    start_time = time.time()
    
    fittedbn = bnbayesfit("bnstruct.csv", "bndata.csv")
    bnsample(fittedbn, 10)
    # learn_from_samples(fittedbn, 10000, "bnstruct.csv")
    
    elapsed_time = time.time() - start_time    
    print "Total execution time:", elapsed_time, "seconds."
