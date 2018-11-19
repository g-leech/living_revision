"""
    Helpers for linear regression.
    Written for understanding, not use.
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

import numpy.polynomial.polynomial as poly



def ols_slope_estimator(xs, ys) :
    n = len(ys)
    sumProducts = sum(ys * xs)
    productSums = sum(xs) * sum(ys)
    numer = n*sumProducts - productSums
    
    squaredSumX = sum(xs)**2
    sumSquaresX = sum(xs**2)
    denom = n*sumSquaresX - squaredSumX
    
    return numer / denom


def ols_intercept_estimator(xs, ys, estSlope=None) :
    averageResponse = np.mean(ys)
    averageExplanatory = np.mean(xs)
    
    if not estSlope :
        estSlope = ols_slope_estimator(xs, ys)
    
    return averageResponse - estSlope * averageExplanatory
    

def simple_sample_error_variance(xs, ys, a, b) :
    preds = [(a + b*x) for x in xs]
    errors = ys - preds
    sse = sum(errors**2)
    # Corrected for two parameters a and b:
    n = (len(xs) - 2)
    
    return  sse / n


def simple_linear_regression_params(xs, ys) :
    beta = ols_slope_estimator(xs, ys)
    alpha = ols_intercept_estimator(xs, ys, beta)
    var = simple_sample_error_variance(xs, ys, alpha, beta)
    
    return alpha, beta, var


def simple_linear_regression(xs, ys) :
    model = sm.OLS(ys, sm.add_constant(xs))
    results = model.fit()
    
    return results.summary()


# S_xx
def sum_of_squared_deviations_from_mean(xs) :
    mu = np.mean(xs)
    squareDeviations = (xs - mu)**2
    
    return sum(squareDeviations)
    

def population_intercept_variance(xs, var) :
    n = len(xs)
    samplingVar = var / n
    meanSquareX = np.mean(xs)**2
    sxx = sum_of_squared_deviations_from_mean(xs)
    
    return samplingVar * (1 + n*meanSquareX/sxx)


def population_slope_variance(xs, var) :
    sxx = sum_of_squared_deviations_from_mean(xs)
    
    return var / sxx
    
# Not very useful: rely on known pop var \sigma^2



def sum_squares_total(ys) :
    muY = np.mean(ys)
    return sum((ys - muY)**2)


def sum_squares_residual(xs, ys, a, b) :
    muY = np.mean(ys)
    preds = np.array([(a + b*x) for x in xs])
    return sum((muY - preds)**2)


def variance_ratio(xs, ys, a, b, numFactors=1) :
    sumSquareTotal = sum_squares_total(ys)
    sumSquareRegression = sum_squares_residual(xs, ys, a, b)
    #sumSquareResiduals = sumSquareTotal - sumSquareResiduals
    meanSquareReg = sumSquareRegression / numFactors
    meanSquareResiduals = simple_sample_error_variance(xs, ys, a, b)
    
    return meanSquareReg / meanSquareResiduals


# estimated standard error of observations
def standard_error_obs(xs, ys, aHat, bHat) :
    meanSquareResiduals = simple_sample_error_variance(xs, ys, aHat, bHat)
    
    return np.sqrt(meanSquareResiduals)


def standard_error_beta(xs, ys, aHat, bHat) :
    s = np.sqrt( simple_sample_error_variance(xs, ys, aHat, bHat) )
    sxx = sum_of_squared_deviations_from_mean(xs)
    
    return s / np.sqrt(sxx)



# VF = F stat
def f_test_pval(varRatio, df1, n):
    return 1 - stats.f.cdf(varRatio, df1, n-2)


# bHat / s * sqrt(sxx) = t stat
def t_test_pval(xs, b, n) :
    t = t_of_null_beta(bHat=b, xs=xs)
    return 1 - stats.t.cdf(t, n-2)


# Test stat for zero mean hypothesis
def t_of_null_beta(bHat, xs) :
    sxx = sum_of_squared_deviations_from_mean(xs)
    s2 = simple_sample_error_variance(xs, ys, a, b)
    
    return bHat / (np.sqrt(s2) / np.sqrt(sxx))


# Interval estimation for simple lines:
def beta_interval(estimate, n, s, sxx, delta) :
    t = stats.t.ppf(1 - delta/2, n-2)
    stderr = s / np.sqrt(sxx)
    intval = t * stderr
    
    return estimate - intval, estimate + intval


def intercept_stderr(xs, s2) :
    n = len(xs)
    muXSquare = np.mean(xs)**2
    sxx = sum_of_squared_deviations_from_mean(xs)
    
    return np.sqrt( s2 * (1/n + muXSquare / sxx) )

    
def alpha_interval(xs, a, s2, delta) :
    stderr = intercept_stderr(xs, s2)
    t = stats.t.ppf(1 - delta/2, n-2)
    
    return a - t*stderr, a + t*stderr



def fit_and_hedge(xs, ys, delta=0.05) :
    a, b = poly.polyfit(xs, ys, deg=1)
    s2 = simple_sample_error_variance(xs, ys, a, b)
    sxx = sum_of_squared_deviations_from_mean(xs)
    
    print("alpha: ", alpha_interval(xs, a, s2, delta))
    betas = beta_interval(b, len(xs), np.sqrt(s2), sxx, delta) 
    print("beta: ", betas )
    
    return betas
    
    

def prediction_stderr(x, xs, s, futureVar=False) :
    n = len(xs)
    muX = np.mean(xs)
    squareDeviance = (x - muX)**2
    sxx = sum_of_squared_deviations_from_mean(xs)
    
    correction = 1 if futureVar else 0
    
    return s * np.sqrt( squareDeviance / sxx + 1/n + correction)

    
# takes into account the sampling variability inherent in estimation of line: 
# gives a range of plausible values for the mean of Y_0
def confidence_interval_for_the_mean(aHat, bHat, x, xs, ys, delta=0.05) :
    n = len(xs)
    s2 = simple_sample_error_variance(xs, ys, a, b)
    stderr = prediction_stderr(x, xs, np.sqrt(s2))
    t = stats.t.ppf(1 - delta/2, n-2)
    intval = t * stderr
    
    point = aHat + bHat * x
    
    return point - intval, point + intval


# but does  not take into account the fact that Y 0  varies about its mean value

# tiny bit wider.
# takes into account both types of variability, so that it provides a range of plausible values for the value that will be observed.
def prediction_interval(aHat, bHat, x, xs, delta=0.05) :
    n = len(xs)
    muX = np.mean(xs)
    squareDeviance = (x - muX)**2
    s2 = simple_sample_error_variance(xs, ys, a, b)
    sxx = sum_of_squared_deviations_from_mean(xs)
    stderr = np.sqrt( squareDeviance / sxx + 1/n + 1)
    t = stats.t.ppf(1 - delta/2, n-2)
    intval = t * np.sqrt(s2) * stderr
    
    point = aHat + bHat * x
    
    return point - intval, point + intval


def residual_standard_error(resids, nParams=2) :
    rss = sum([r**2 for r in resids ])
    n = len(resids)
    return np.sqrt(rss / (n-nParams))


def standardise_residuals(resids):
    return resids / residual_standard_error(resids)


def residual_analysis(df, x, y, model) :
    preds = model(df[x])
    residuals = df[y] - preds
    stdResiduals = residuals / residual_standard_error(residuals)
    
    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    axs[0][0].scatter(df[x], df[y])
    axs[0][0].plot(df[x], preds)
    axs[0][0].set_xlabel(x), axs[0][0].set_ylabel(y)
    plt.suptitle("                Raw plus fit / Response v Residuals \n Explan v Residuals / Residual hist")
    
    axs[0][1].scatter(df[y], stdResiduals)
    xRange = np.arange(df[y].min(), df[y].max(), df[y].max()/10)
    axs[0][1].plot(xRange, [0]*len(xRange), linestyle="--", color="gray")
    plt.xlabel(y), plt.ylabel("residual")
    
    sns.residplot(df[x], df[y], lowess=True, ax=axs[1][0])
    plt.ylabel("residual")
    
    sns.distplot(stdResiduals, ax=axs[1][1])
    axs[1][1].set_xlabel("residual")
    plt.show()
    
    fig, axs = plt.subplots(figsize=(5,5))
    stats.probplot(stdResiduals, plot=plt)
    plt.show()


# Slope estimators are ~ N, so difference is ~ N
# Var of difference = var slope1 + var slope2
# True if difference is more than 2 stddevs from 0
def crudely_reject_slope_equality(xs1, ys1, xs2, ys2, \
                              alpha1, alpha2, \
                              beta1, beta2) :
    se1 = standard_error_beta(xs1, ys1, alpha1, beta1)
    se2 = standard_error_beta(xs2, ys2, alpha2, beta2)
    diffVar = se1**2 + se2**2
    diff = beta1 - beta2
    
    return abs(diff) > 1.96 * np.sqrt(diffVar)



# Roughly right
def pearson_r(X, Y) :
    muX, muY = np.mean(X), np.mean(Y)
    sX, sY = np.std(X), np.std(Y)
    
    xErrors, yErrors = X - muX, Y - muY
    sumOfPairedErrorProducts = np.sum( xErrors * yErrors )
    besselN = len(X) - 1
    covariance = sumOfPairedErrorProducts / besselN
    jointStd = sX * sY

    return covariance / jointStd
