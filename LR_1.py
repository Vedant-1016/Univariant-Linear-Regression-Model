import numpy as np
import matplotlib.pyplot as plt

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

def compute_cost(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

tmp_f = compute_cost(x,w,b)
plt.plot(x,tmp_f,label='Prediction')    
plt.scatter(x,y,marker='x',color='red',label="Training Data")
plt.title("Linear Regression Fit")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend()
plt.show()
