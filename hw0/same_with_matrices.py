import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df= pd.read_csv("C:/Users/Edo/Desktop/ML/HW0/linear_regression_data.csv")
def run_regression (x,y):
    X=np.array([[1,x] for x in x])
    Y=np.array(y)
    beta= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return beta
beta= run_regression(list(df['X']),list(df['Y']))
line_x = range(-20, 80)
line_y = [beta[0]+beta[1]*x for x in line_x]
# Plot the intercept
plt.clf()  # Clears the figure from previous output.
plt.scatter(df['X'], df['Y'])
plt.plot(line_x, line_y, color='red')
# Save to a file with your name
plt.show()

    


