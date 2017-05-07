import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Edo/Desktop/linear_regression_data.csv")
def mean(numbers):
    return (sum(numbers)) / len(numbers)
def run_regression ( x , y ):
    "Finding linear regression coefficients "
    sum_xy=0
    for i in range(len(x)):
        sum_xy=sum_xy+x[i]*y[i]
    mean_xy= sum_xy/len(x)
    sum_xx=0
    for i in range(len(x)):
        sum_xx= sum_xx+x[i]*x[i]
    mean_xx=sum_xx/len(x)
    b1 = (mean_xy-mean(x)*mean(y))/(mean_xx-(mean(x))*mean(x))
    b0 = mean(y)-b1*mean(x)
    return b0,b1

b0, b1 = run_regression(list(df['X']), list(df['Y']))
line_x = range(-20, 80)
line_y = [b0 + b1 * x for x in line_x]
# Plot the intercept
plt.clf()  # Clears the figure from previous output.
plt.scatter(df['X'], df['Y'])
plt.plot(line_x, line_y, color='red')
plt.savefig("homework0.eduard.png", dpi=320)
