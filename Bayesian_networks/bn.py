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

import numpy as np

def read_data_file(input_file):
    """Read a csv data file and produce a numpy ndarray. 
    0s and 1s are expected. 
    """
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
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
    condition_str = ""
    for condition, bool_value in conditions: 
        condition_str = condition_str + str(condition) + " = " + str(bool_value) + "; "
    condition_str = condition_str[:-2]
    print "Probability(", var_index, " = ", 1, "|", condition_str, ") =", probability
    print "Probability(", var_index, " = ", 0, "|", condition_str, ") =", 1 - probability

def calculate_conditional_prob(var_index, cond_list, data, alpha=1., beta=1.):
    """Calculate conditional probabilities for a variable.
    """
    rows = data.shape[0]
    # Output is an ordered dictionary, containing probability for every
    # condition combination. 
    output = OrderedDict()
    # Generate all combinations for values of conditions
    var_combinations = list(itertools.product([0, 1], repeat=len(cond_list)))
    for combination in var_combinations: 
        count_1_occurances = 0
        count_0_occurances = 0
        values = zip(cond_list, combination)  # possible values for combinations
        for row in range(rows): 
            success = True
            for condition, bool_value in values: 
                if data[row][condition] != bool_value: 
                    success = False
                    break
            if success: 
                if data[row][var_index] == 1: 
                    count_1_occurances += 1
                else: 
                    count_0_occurances += 1
        # Calculate probability 
        probability = (alpha + float(count_1_occurances)) / (alpha + beta + float(count_1_occurances + count_0_occurances))
        output[tuple(values)] = probability
        print_probability(values, probability, var_index)
    return output
    
def estimate_parameter(structure, data, index):    
    """Estimate parameter at index. 
    Uses a bayesian approach. Implemented as described in section
    9.4 of BRML book (page 199). 
    """
    rows = data.shape[0]
    conditions = read_conditions(structure, index)
    if not conditions:  # if target variable has no conditions calculate directly
        total = np.sum(data[:, index])  # count 1s at index
        prob = (1 + float(total)) / (2 + float(rows))
        print "Probability(", index, " = 0 ) =", 1 - prob
        print "Probability(", index, " = 1 ) =", prob
        return [prob]
    else: 
        return calculate_conditional_prob(index, conditions, data)

def sample(prob):
    """"""
    rand = random.random()
    if rand < prob: 
        return 1
    return 0

def conditional_sample(key, variable, current_dict):
    output = current_dict
    remaining = []
    for var, value in variable.items(): 
        found = True
        for rel in var: 
            if rel[0] in output: 
                if output[rel[0]] != rel[1]: 
                    found = False
                    break
            else: 
                if not key in remaining: 
                    remaining.append(key)
                found = False
                break
        if found: 
            # print output
            output[key] = sample(value)
            # print "key:", key, "var:", var, "value:", value, "sample:", output[key]
    return output, remaining
            
def ancestral_sampling(network):
    """Implements ancestral sampling as described in slide 8 of the MLAP 
    Markov Chain Monte Carlo lecture."""
    # Sort by length
    network = dict(sorted(network.iteritems(), key=lambda x: len(x[1])))
    output = {}
    remaining = []
    for key, item in network.items(): 
        if isinstance(item, list): 
            output[key] = sample(item[0])
        elif isinstance(item, OrderedDict):
            output, rem2 = conditional_sample(key, item, output)
            remaining = remaining + rem2
    for item in remaining: 
        output, rem2 = conditional_sample(item, network[item], output)
    return output

def ancestral_sampling_r(network, predefined=None):
    if predefined: 
        output = predefined
    else: 
        output = {}
    remaining = {}
    for key, item in network.items(): 
        if isinstance(item, list): 
            output[key] = sample(item[0])
            # print "key:", key, "sample:", output[key]
        elif isinstance(item, OrderedDict):
            output, rem2 = conditional_sample(key, item, output) 
            print "o, rem2", output, rem2           
            remaining = dict(remaining.items() + rem2.items())
    print "Remaining:", remaining
    if remaining: 
        output = dict(output.items() + ancestral_sampling(remaining, output).items())
    # for item in remaining: 
    #    output, rem2 = conditional_sample(item, network[item], output)
    return output
    
def bnbayesfit(structure_file_name, data_file_name):
    """Estimate parameters of bayesian network, defined in structure_file_name
    by using data from data_file_name."""
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
    """"""
    output = np.empty((nsamples, len(fittedbn)), dtype=np.int)
    for i in range(nsamples): 
        sample_dict = ancestral_sampling(fittedbn) 
        output[i] = np.array([sample_dict[key] for key in sorted(sample_dict.keys())])
    print output
    return output
    
def learn_from_samples(fittedbn, nsamples, structure_file_name):
    samples = bnsample(fittedbn, nsamples)
    fit2 = bnbayesfit(structure_file_name, samples)

if __name__ == '__main__':
    fittedbn = bnbayesfit("bnstruct.csv", "bndata.csv")
    bnsample(fittedbn, 10)
    # learn_from_samples(fittedbn, 10000, "bnstruct.csv")
