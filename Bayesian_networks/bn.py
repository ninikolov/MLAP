'''
Created on 25 Feb 2014

@author: Y6189686
'''

import csv
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx 

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
    
def bnbayesfit(structure_file_name, data_file_name):
    structure = read_data_file(structure_file_name)
    data = read_data_file(data_file_name)
    

if __name__ == '__main__':
    print read_data_file("bndata.csv")