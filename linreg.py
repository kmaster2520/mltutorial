from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

'''
y = mx+b solve for m and b:
E = expected value/mean
m = covariance(x,y) / variance(x) (slope)
b = Ey - m * Ex (y-intercept)

SE = squared error
r2 = r-squared = 1 - SE(y-hat) / SE(y-bar)



'''

def best_fit_slope_and_intercept(xs, ys):
    ex = mean(xs)
    ey = mean(ys)
    cov = mean(xs*ys) - ex*ey
    var = mean(xs*xs) - ex*ex
    m = 1.0 * cov / var
    b = ey - m * ex
    return m, b

def squared_error(ys_orig, ys_line):
    return sum( (ys_line - ys_orig)**2 )

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1.0 - (squared_error_regr / squared_error_y_mean)

m, b = best_fit_slope_and_intercept(xs, ys);
#print(m,b)

regression_line = [(m*x)+b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()





