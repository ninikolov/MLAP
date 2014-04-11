'''
Created on 9 Apr 2014

@author: nikola
'''

def class_prob(n, data):
    """Naive bayes?"""
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
    """MLE?"""
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


    
def bnbayesfit2(structure_file_name, data_file_name):
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