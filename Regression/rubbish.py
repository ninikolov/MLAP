'''
Created on 14 Mar 2014

@author: nikola
'''

def test_logistic():
    x, y = read_data_file('stock_price.csv', True)  
    x = feature_selection_financial_data_logistic(x)
    
    train_x = x[0 : len(x) / 2]
    train_y = y[0 : len(y) / 2]
    validate_x = x[len(x) / 2 : len(x)]
    validate_y = y[len(y) / 2 : len(y)]
    train_x_n = normalize_data(train_x)
    validate_x_n = normalize_data(validate_x)
    
    logreg = linear_model.LogisticRegression()
    logreg.fit(validate_x, validate_y)
    print logreg.get_params()
    print logreg.score(train_x, train_y)

def identity(class1, class2):
    if class1 == class2: 
        return 1
    return 0

def test_classess(theta, data, y, points_to_test=50):
    for i in range(points_to_test): 
        print "\nData Point ", i
        
        for j in range(len(theta)): 
            print "Class ", j , " with prob ", class_probability(theta, data[i], j), " | Actual: ", y[i]

def visualize_bn(network):
    G = nx.DiGraph()
    for i in range(len(network)):
        for n in range(len(network[i])):  
            if network[i][n] == 1: 
                G.add_edges_from([(i, n)])
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    plt.show()   

def num(number):
    '''Convert a string to a number.'''
    try:
        return int(number)
    except ValueError:
        return float(number)
    
def read_file4(input_file):
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        X = np.ones(shape=(len(rows) - 10, 21))
        Y = np.zeros(len(rows) - 10)
        for i in range(0, len(rows)):
            if i < 10: 
                continue
            for n in range(1, 11): 
                X[i - 10][n - 1] = rows[i - n ][0]
            for n in range(11, 21): 
                X[i - 10][n - 1] = rows[i - (n - 10) ][1]
            Y[i - 10] = compute_class(float(rows[i][1]), float(rows[i - 1][1]))
            if i < 20 and DEBUG: 
                current_price = float(rows[i][1])
                previous_price = float(rows[i - 1][1])
                change = current_price - previous_price
                percentage = (100 * change) / previous_price
                print i, "Current: ", current_price, " Previous: ", previous_price, " Chage: ", percentage
    print X[0], Y[0], Y[1]
    print "Last:", X[len(X) - 1], Y[len(Y) - 1]
    return X, Y


def predict_class(theta, target, x):
    # m, n = x.shape
    # p = np.zeros(shape=(m, 1))  
    
    theta_x = np.dot(theta, x)
    h = np.exp(theta_x)
    probabilities = h / np.sum(h, axis=0)
 
    predictions = np.zeros((len(x), 1))
    predictions[:, 0] = np.argmax(probabilities, axis=0)
    return predictions

def sigmoid(X):
    '''Compute the sigmoid function '''
    # d = zeros(shape=(X.shape))
 
    den = 1.0 + np.exp (-1.0 * X)
 
    d = 1.0 / den
 
    return d





def compute_cost(theta, X, y, regularise=False):
    """21.04"""
    g = True
    theta = theta.reshape((5, np.shape(X)[1]))
    #accuracy(theta, X, y)
    m = X.shape[0] 
    n = theta.shape[0]  
    loss = 0.
    grad = np.zeros(shape=(n, np.shape(X)[1]))
    for i in range(0, m):        
        current_class = int(y[i])
        probability = class_probability(theta, X[i], current_class)
        loss += np.log(probability)
        if g: 
            for k in range(0, n):
                grad[k] += np.dot(X[i], (identity(current_class, k) - probability))
    # print "Output: ", -loss, -grad.flatten() 
    if not g: 
        if regularise: 
            return -(LAMBDA * loss - (1 - LAMBDA) * np.sum([abs(th) ** 2 for th in theta.flatten()]))
        else: 
            return -loss
    return [-loss, grad.flatten()]



def compute_cost(theta, X, y):
    """Original one"""
    theta = theta.reshape((5, np.shape(X)[1]))
    # print theta
    m = X.shape[0] 
    n = theta.shape[0]
    loss = 0.
    # grad = np.zeros(theta.shape)
    # For each data point 
    for i in range(0, m): 
        # Vector of dot products for every theta
        dot_vector = np.array([np.dot(theta[j], X[i]) for j in range(0, n)])
        lse = logsumexp(dot_vector)
        # Dot product of currently looked class and data point
        theta_current_x = np.dot(theta[int(y[i])], X[i])
        loss += -(theta_current_x - lse)
        """
        for k in range(0, n):
            prob = class_probability(theta, X[i], k)
            #print prob
            grad[k] = -(np.dot(X[i], (identity(y[i], k) - prob)))
        """
    print "Output: ", loss
    return loss


def log_sum_exp(Vector):
    # first find max of the sequence
    Max = Vector[0]
    for LP in Vector:
        if (LP > Max): 
            Max = LP
    Sum = 0.0
    for LP in Vector:
        Sum += np.exp(LP - Max)    
    jazz = (Max + np.log(Sum))
        
    return jazz

def grad_lr(theta, X, y):
    m = X.shape[0] 
    n = theta.shape[0]
    grad = np.zeros(theta.shape)
    for i in range(0, m): 
        for k in range(0, n):
            prob = class_probability(theta, X[i], k)
            # print prob
            grad[k] = -(np.dot(X[i], (identity(y[i], k) - prob)))
    # print "Grad: ", grad.flatten()
    return grad


def compute_cost2(theta, X, y):
    theta = theta.reshape((5, np.shape(X)[1]))
    # print theta
    m = X.shape[0] 
    n = theta.shape[0]
    loss = 0   
    # theta_x = np.dot(theta, X) 
    theta_x = np.array(np.dot(theta, X[i]) for i in range(0, m))
    print theta_x
    hypothesis = np.exp(theta_x)      
    probabilities = hypothesis / np.sum(hypothesis, axis=0)
    lse = np.log(probabilities)
    
def compute_cost(theta, X, y): 
    theta = theta.reshape((5, np.shape(X)[1]))
    # print theta
    m = X.shape[0] 
    n = theta.shape[0]
    calculated_log = np.sum(np.array([log_sum_exp(np.dot(X, theta[i])) for i in range(0, n)]))   
    loss = 0
    for i in range(0, n): 
        loss += np.sum(np.dot(X, theta[i]) - calculated_log) 
    # loss = np.sum((np.dot(X, theta[i]) - calculated_log) for i in range(n))
    print -loss, theta.flatten()
    return -loss
    # return J[0][0]

def read_file3(input_file):
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = list(reader)
        X = np.ones(shape=(len(rows) - 10, 2))
        Y = np.zeros(len(rows) - 10)
        for i in range(0, len(rows)):
            if i < 10: 
                continue
            X[i - 10][0] = rows[i][2]                  
            Y[i - 10] = rows[i][1]
    print X[0], Y[0], Y[1]    
    return X, Y

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
    Data = np.array([np.ones(len(Y)), X])
    Y = np.array(Y)
    
    return Data.T, X, Y

def linear_old(input_filename):
    Data, X, Y = read_file(input_filename)  
    
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, len(Y)), Y, 'ro')

    ax2 = ax1.twinx()
    
    ax2.plot(range(0, len(X)), X, 'bo')
    
    # plt.plot(range(0, len(Y)), Y, 'o', range(0, len(X)), X, 'ro')
    plt.show()
    
def linear_numpy(input_filename):
    Data, X, Y = read_file(input_filename)    
    w = np.linalg.lstsq(Data, Y)[0]  
    print(w)
    line = w[1] * X + w[0]
    plt.plot(X, line, 'r-', X, Y, 'o')
    plt.show()
    
    
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




################################################


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
    sq_error = mean_squared_loss(w, x, y)
    print sq_error
    line = np.zeros(len(y))
    for i in range(len(x[0])): 
        line += w[i] * x[:, i]
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()
    
def linear2():
    x, y = read_data_file('stock_price.csv')
    x_print = np.array(np.arange(0, len(y)))
    m, n = np.shape(x)
    print m, n
    max_iterations = 200
    theta = np.ones(n)
    print theta
    theta = fmin_bfgs(mean_squared_loss, theta, args=(x, y), maxiter=max_iterations)
    print theta
    line = np.zeros(len(y))
    for i in range(n): 
        line += theta[i] * x[:, i]
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()
    
    
def linear_numpy2(input_filename):
    x, y = read_file3(input_filename)
    # x = feature_selection_financial_data(x)
    x_print = np.array(np.arange(0, len(y)))
    w = np.linalg.lstsq(x, y)[0]  
    print(w)
    sq_error = mean_squared_loss(w, x, y)
    print sq_error
    line = np.zeros(len(y))
    for i in range(len(x[0])): 
        line += w[i] * x[:, i]
    plt.plot(x_print, line, 'r-', x_print, y, 'o')
    plt.show()
    
    