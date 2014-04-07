'''
Created on 14 Mar 2014

@author: nikola
'''

def num(number):
    '''Convert a string to a number.'''
    try:
        return int(number)
    except ValueError:
        return float(number)

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
    x, y = read_file2('stock_price.csv')
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