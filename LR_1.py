import numpy as np
import matplotlib.pyplot as plt
import math

x = np.array([1.0 , 2.0])
y = np.array([300.0, 500.0])
m = len(x)
print(f"Number of training examples : {m}")
i = 0
x_i = x[i]
y_i = y[i]

#plot the data
plt.scatter(x,y,marker='x',color='red')
plt.title("Scatter plot of training data")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

w = 200
b = 100

def lr_algo(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

def cost_fxn(x,y,w,b):
    m = x.shape[0]
    total_cost=0
    for i in range(m):
        f_wb_i = w * x[i] + b
        total_cost += (f_wb_i - y[i])**2
    total_cost = total_cost / 2*m
    return total_cost

def gradient_descent(x,y,w_in,b_in,alpha,num_iters):
    w_in=0
    b_in=0
    m = x.shape[0]
    for i in range(num_iters):
        dj_dw = 0
        dj_db = 0
        for j in range(m):
            f_wb_j = w_in * x[j] + b_in
            dj_dw += (f_wb_j - y[j])*x[j]
            dj_db += (f_wb_j - y[j])
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
    return w_in,b_in

tmp_f = lr_algo(x,w,b)
plt.plot(x,tmp_f,label='Prediction')    
plt.scatter(x,y,marker='x',color='red',label="Training Data")
plt.title("Linear Regression Fit")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend()
plt.show()
