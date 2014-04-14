'''
Created on 25 Feb 2014

Code for MLAP Open Assessment Part 2. 
Bayesian networks. 

@author: Y6189686
'''

import csv
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx 
import itertools
from collections import OrderedDict
import random
from numpy import ndarray

def read_data_file(input_file):
    """Read a csv data file and produce a numpy ndarray. 
    0s and 1s are expected. 
    """
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        data = np.zeros((len(rows), len(rows[0])), dtype=np.int)
        for i in range(0, len(rows)):
            # print rows[i]
            data[i] = rows[i]    
    return data
    
def read_conditions(structure, index):
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

def calculate_conditional_prob(var_index, cond_list, data):
    """"""
    rows, cols = data.shape
    output = OrderedDict()
    # Generate all combinations for values of conditions
    var_combinations = list(itertools.product([0, 1], repeat=len(cond_list)))
    for combination in var_combinations: 
        count = 0
        count_not = 0
        values = zip(cond_list, combination)
        for row in range(rows): 
            success = True
            for condition, bool_value in values: 
                if data[row][condition] != bool_value: 
                    success = False
                    break
            if success: 
                if data[row][var_index] == 1: 
                    count += 1
                else: 
                    count_not += 1
        probability = (1. + float(count)) / (2. + float(count + count_not))
        output[tuple(values)] = probability
        print_probability(values, probability, var_index)
    return output
    
def estimate_parameter(structure, data, index):    
    """Bayesian parameter estimate"""
    rows, cols = data.shape
    total = np.sum(data[:, index])
    conditions = read_conditions(structure, index)
    if not conditions: 
        prob = (1 + float(total)) / (2 + float(rows))
        print "Probability(", index, " = 0 ) =", 1 - prob
        print "Probability(", index, " = 1 ) =", prob
        return [prob]
    else: 
        return calculate_conditional_prob(index, conditions, data)

def sample(prob):
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
    output = {}
    remaining = []
    for key, item in network.items(): 
        if isinstance(item, list): 
            output[key] = sample(item[0])
            # print "key:", key, "sample:", output[key]
        elif isinstance(item, OrderedDict):
            output, rem2 = conditional_sample(key, item, output)
            remaining = remaining + rem2
    # print "Remaining:", remaining
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
    structure = read_data_file(structure_file_name)
    if not isinstance(data_file_name, ndarray): 
        data = read_data_file(data_file_name)
    else: 
        data = data_file_name
    rows, cols = data.shape
    print rows, cols
    fittedbn = {}
    # print "Theta: ", theta
    for i in range(cols): 
        fittedbn[i] = estimate_parameter(structure, data, i)
    # sort
    fittedbn = dict(sorted(fittedbn.iteritems(), key=lambda x: len(x[1])))
    print fittedbn    
    return fittedbn
    
def bnsample(fittedbn, nsamples):
    """"""
    output = np.empty((nsamples, len(fittedbn)), dtype=np.int)
    for i in range(nsamples): 
        sample = []
        d = ancestral_sampling(fittedbn) 
        # print d
        for key in sorted(d.keys()): 
            output[i][key] = d[key]
        # output.append(sample)
    print output
    return output
    
def learn_from_samples(fittedbn, nsamples, structure_file_name):
    samples = bnsample(fittedbn, nsamples)
    fit2 = bnbayesfit(structure_file_name, samples)

if __name__ == '__main__':
    print read_data_file("bndata.csv")
    fittedbn = bnbayesfit("bnstruct.csv", "bndata.csv")
    bnsample(fittedbn, 10)
    learn_from_samples(fittedbn, 10000, "bnstruct.csv")
