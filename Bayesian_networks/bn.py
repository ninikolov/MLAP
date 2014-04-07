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

def read_data_file(input_file):
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        data = np.zeros((len(rows), len(rows[0])), dtype=np.int)
        for i in range(0, len(rows)):
            # print rows[i]
            data[i] = rows[i]    
    return data

def visualize_bn(network):
    G = nx.DiGraph()
    for i in range(len(network)):
        for n in range(len(network[i])):  
            if network[i][n] == 1: 
                G.add_edges_from([(i, n)])
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    plt.show()
    
def class_prob(n, data):
    rows, cols = data.shape
    prob_true = 1.
    prob_false = 1.
    total_class = np.sum(data[:, n])
    print "Estimating parameter", n, data.shape, total_class
    for col in range(0, cols): 
        if col == n: 
            continue
        else: 
            count_true = 0
            count_false = 0
            for row in range(0, rows): 
                if data[row][n] == 1: 
                    count_true += data[row][col]
                else: 
                    count_false += data[row][col]
            print "counts ", count_true, count_false, "total: ", np.sum(data[:, col])
            prob_true = prob_true * (float(count_true) / float(total_class))
            prob_false = prob_false * (float(count_false) / float(rows - total_class))            
            print "probs ", prob_true, prob_false
    
    prob_true = prob_true * float(total_class) / float(rows)
    prob_false = prob_false * (float(rows - total_class) / float(rows))
    return prob_true / (prob_true + prob_false)
    
def estimate_parameter(structure, data, index):    
    theta = np.empty(np.sum(structure[:, index]) ^ 2)
    rows, cols = data.shape
    conditions = []
    total = np.sum(data[:, index])
    for item in range(structure.shape[1]): 
        if structure[item][index] == 1:
            conditions.append(item)
    if not conditions: 
        return [total / rows, (1 - total / rows)]
    else: 
        output = []
        combinations = list(itertools.product([0, 1], repeat=len(conditions)))
        for prob in (0, 1): 
            for combination in combinations: 
                values = zip(conditions, combination)
                count = 0
                for row in range(rows): 
                    if data[row][index] == prob: 
                        success = True
                        for i, b in values: 
                            if data[row][i] != b: 
                                success = False
                        if success: 
                            count += 1
                if prob == 1: 
                    probability = float(count) / float(total)
                else: 
                    probability = float(count) / float(rows - total)
                output.append(probability)
                condition_str = ""
                for i, b in values: 
                    condition_str = condition_str + str(i) + " = " + str(b) + ","
                condition_str = condition_str[:-1]
                print "Probability(", index, " = ", prob, "|", condition_str, ") =", probability
    # print output
    return output

def estimate_parameter2(structure, data, index):    
    rows, cols = data.shape
    conditions = []
    total = np.sum(data[:, index])
    for item in range(structure.shape[1]): 
        if structure[item][index] == 1:
            conditions.append(item)
    if not conditions: 
        prob = (1 + float(total)) / (2 + float(rows))
        print "Probability(", index, " = 0 ) =", 1 - prob
        print "Probability(", index, " = 1 ) =", prob
        return [prob]
    else: 
        output = OrderedDict()
        combinations = list(itertools.product([0, 1], repeat=len(conditions)))
        for prob in (1,): 
            for combination in combinations: 
                values = zip(conditions, combination)
                # print "Values:", values
                count = 0
                count_not = 0
                for row in range(rows): 
                    success = True
                    for i, b in values: 
                        if data[row][i] != b: 
                            success = False
                            break
                    if data[row][index] == prob and success: 
                        count += 1
                    elif success: 
                        count_not += 1
                probability = (1 + float(count)) / (2 + float(count + count_not))
                # print "Test:", OrderedDict([(a, b) for a, b in zip(combination, probability)])
                # output.append(OrderedDict([(combination, probability)]))
                output[tuple(values)] = probability
                condition_str = ""
                for i, b in values: 
                    condition_str = condition_str + str(i) + " = " + str(b) + "; "
                condition_str = condition_str[:-2]
                print "Probability(", index, " = ", prob, "|", condition_str, ") =", probability
    return output

def sample(prob):
    rand = random.random()
    if rand < prob: 
        return 1
    return 0
            
def ancestral_sampling(network):
    output = {}
    remaining = []
    #print len(network)
    for key, item in network.items(): 
        if isinstance(item, list): 
            output[key] = sample(item[0])
        elif isinstance(item, OrderedDict): 
            for var, value in item.items(): 
                found = True
                for rel in var: 
                    if rel[0] in output: 
                        if output[rel[0]] != rel[1]: 
                            found = False
                    else: 
                        remaining.append(rel[0])
                if found: 
                    output[key] = sample(value)
                    print "var:", var, "value:", value, "sample:",output[key]
        #print output 
    for item in remaining: 
        dict = network[item]
        for var, value in dict.items(): 
            #print var
            found = True
            for rel in var: 
                if rel[0] in output: 
                    if output[rel[0]] != rel[1]: 
                        found = False
                else: 
                    remaining.append(rel[0])
            if found: 
                output[key] = sample(value)
    #print output
    return output
    
def bnbayesfit2(structure_file_name, data_file_name):
    structure = read_data_file(structure_file_name)
    data = read_data_file(data_file_name)
    rows, cols = data.shape
    print rows, cols
    theta = OrderedDict() 
    # print "Theta: ", theta
    for i in range(cols): 
        theta[i] = estimate_parameter2(structure, data, i)
    # sort
    theta = OrderedDict(sorted(theta.iteritems(), key=lambda x: len(x[1])))
    print theta
    bnsample(theta, 10)
    
def bnsample(fittedbn, nsamples):
    output = []
    for i in range(nsamples): 
        sample = []
        d = ancestral_sampling(fittedbn) 
        #print d
        for key in sorted(d.keys()): 
            sample.append(d[key])
        output.append(sample)
    print output
    
def bnbayesfit(structure_file_name, data_file_name):
    structure = read_data_file(structure_file_name)
    data = read_data_file(data_file_name)
    rows, cols = data.shape
    print rows, cols
    theta = OrderedDict() 
    # theta.fill(1 / float(cols))
    print "Theta: ", theta
    for i in range(cols): 
        theta[i] = class_prob(i, data)
    for i in theta: 
        print i
    print theta 

if __name__ == '__main__':
    print read_data_file("bndata.csv")
    bnbayesfit2("bnstruct.csv", "bndata.csv")
