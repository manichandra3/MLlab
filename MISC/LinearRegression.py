import numpy as np
from pandas.core.groupby.groupby import Axis

# simple linear regression with
# one feature and one dependent variable.
# Y = f(x) = W * X + b
# j(W,b) = 1/(2*m) sum[1,m](y'[i]-y[i])^2

def prediction(x, w, b):
    return w * x + b


def cost(x, y, w, b):
    m = x.shape[0]
    return 1/(2*m)*np.sum(prediction(x, w, b),-y)**2


data_set = np.array([[4, 5, 7, 10, 15, 12, 18, 20, 22, 25], [2, 3, 5, 7, 9, 8, 11, 14, 16, 19]])
