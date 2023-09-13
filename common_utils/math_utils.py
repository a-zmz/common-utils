"""
This module contains math related helper functions to process data.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import sympy as sp


def get_auc(y, x_lim, dx=1):
    """
    Get area under the curve (AUC) of some cumulative counts.

    params
    ===
    y: list-like monotonic array.

    x_lim: int, limit of x for auc calculation.

    dx: int, step/delta x for auc calculation.
        Default: 1.

    return
    ===
    auc using trapezoidal rule.
    """
    # get auc
    return np.trapz(y[:x_lim], dx=dx)


def random_sampling(array, size, axis=0, repeats=10000):
    rng = np.random.default_rng()

    # TODO: this is trial level chance, but also need a session level chance, i.e.,
    # randomly choose size from all number of wrong licks within a session, then use
    # that size as the size here to create random samples 
    samples = np.array(
        [rng.choice(array, size=size, axis=axis) for _ in range(repeats)]
    ).astype(int)

    return samples


# TODO: may 10th is this func useful?
def psychometric(x, alpha, beta, gamma, lapse):
    return gamma + (1 - gamma - lapse) * \
            (1 / (1 + np.exp(-beta * (x - alpha))))


def get_logistic_derivative(k, x_0, level):
    """
    Implementation of logicstic function using sympy:
    `f(x) = L / (1 + exp(-k * (x - x_0)))`, and to get its derivative.

    params
    ===
    x: list like monotonic array.

    L: the supremum of the values of the function.

    k: the logistic growth rate or steepness of the curve.

    x_0: the x value of the function's midpoint.

    level: int, level of derivative to get.

    return
    ===
    derivative: sympy function of the derivative.

    np_derivative: numpy function of the derivative.
    """
    # define sympy logistic function
    x = sp.symbols('x')
    func = 1 / (1 + sp.exp(-k * (x - x_0)))

    # get derivative
    derivative = sp.diff(func, x, level)

    # solving second derivative and find max & min
    #critical_points = sp.solve(sec_dev, x, dict=True)

    # NOTE solving derivatives takes way too long, thus convert it into a np func to
    # solve numerically
    # get np compatibles
    np_derivative = sp.lambdify(x, derivative, modules=['numpy'])

    return derivative, np_derivative


def logistic(x, k, x_0):
    #TODO: may 11th define funcs using sympy, and convert to numpy where necessary
    """
    Implementation of logicstic function:
    `f(x) = L / (1 + exp(-k * (x - x_0)))`

    params
    ===
    x: list like monotonic array.

    L: the supremum of the values of the function.
        Default: 1.

    k: the logistic growth rate or steepness of the curve.

    x_0: the x value of the function's midpoint.

    """
    # define logistic function
    # TODO may 22 set L = 1 to make sure the curve does not exceed 1
    return 1 / (1 + np.exp(-k * (x - x_0)))
    #return L / (1 + np.exp(-k * (x - x_0)))


def exponential(x, c, k, x_0):
    """
    Implementation of exponential function:
    `f(x) = c + exp(-k * (x - x_0)))`

    params
    ===
    x: list like monotonic array.

    c: constant.

    k: the logistic growth rate or steepness of the curve.

    x_0: the x value of the function's midpoint.

    """
    # define logistic function
    return c + np.exp(-k * (x - x_0))


def step_function(x, a, b):
    # NOTE: Heaviside step function is NOT differentiable, thus use logistic function
    # makes more sense.
    return np.heaviside(x - a, 0) * b


def fit_linear(x, y, normalise=False):
    """
    Fit data to linear least-squares regression, and get the linear model parameters.

    params
    ===
    x: array like measurement on x-axis.

    y: array like measurement on y-axis.

    return
    ===
    k: float, slope of the regression line.
    b: float, intercept of the regression line.
    r: float, Pearson correlation coefficient of the regression line.
    p: float, p value of for a hypothesis test whose null hypothesis is that the
        slope is zero, using Wald Test with t-distribution of the test statistic.
    """
    if normalise:
        # normalise y
        norm_y = y / y.max()
        # normalise x
        norm_x = x / x.max()
    else:
        norm_y = y
        norm_x = x

    # get results
    results = linregress(
        x=norm_x,
        y=norm_y,
    )
    k, b, r, p, _, = results
    # pred_y = results.intercept + results.slope * x, get p-value by
    # results.pvalue, and R squared results.rvalue ** 2

    return k, b, r, p


def fit_logistic(x, y, best_mid, num_max, normalised=False):
    """
    Fit data to logistic function, get the parameters of the logistic function, and
    find the maxima of its second derivative that indicates the biggest
    acceleration in y.

    params
    ===
    x: list like monotonic array.

    y: list like monotonic array.

    best_mid: int, the best, in theory, midpoint of the logistic function for the
    data.

    num_max: int, maxima that is found numerically.

    normalised: bool, if input data is normalised.
        Default: False.

    return
    ===
    np array: [L, k, x_0, maxima, minima, d_mid], round up to 3 digits.
    """
    if not normalised:
        # normalise y
        norm_y = y / y.max()
        # normalise x
        norm_x = x / x.max()
    else:
        norm_y = y
        norm_x = x

    # fit normalised data to logistic func
    params, covs = curve_fit(
        f=logistic,
        xdata=norm_x,
        ydata=norm_y,
    )

    # solve second derivative to find max & min
    _, np_sec = get_logistic_derivative(*params, 2)
    sec_devs = np_sec(norm_x)
    if np.isnan(sec_devs).all():
        #print(f"\n> Fitted logistic function is not differentiable, most likely",
        #end=" cuz it's almost a step function, finding maxima & minima numerically.")

        # midpoint is the place with highest difference
        mid_idx = np.argsort(np.diff(y))[-1]
        # maxima is assigned numerically
        maxima = num_max
        # minima is 2cm after midpoint
        try:
            minima = x[mid_idx + 2]
        except IndexError:
            minima = x[mid_idx + 1]
    else:
        # highest acceleration
        maxima = x[np.argmax(sec_devs)]
        # highest decceleration
        minima = x[np.argmin(sec_devs)]

    # get distance in cm between fitted midpoint to best midpoint
    d_mid = params[-1] * x.max() - best_mid

    # get predicted y
    # pred_y = logistic(norm_x, *params)
    # y_mid = logistic(x_0, *params)

    # plot1: x- normalised tunnel, overlay tunnel, y ratio lick, get mean logistic params
    # and mean starting location, fit logi, darker colour; plot 2 licks
    # plot sympy func too:
    #p1 = sp.plotting.plot((func, fs_dev), (data, 0, 1), show=False)
    #p1 = sp.plotting.plot(fs_dev, data, show=False)
    #p2 = sp.plotting.plot(sec_dev, (data, 0, 1), show=False)
    #p1.extend(p2) # add second plot to the first
    # plot sns.lineplot(x=x, y=pred_y) & (x=x, y=norm_y)

    return np.append(params, [maxima, minima, d_mid]).round(3)


def fit_heavistep(x, y, best_mid, normalised=False):
    """
    Fit data to exponential function, get the parameters of the logistic function, and
    find the maxima of its second derivative that indicates the biggest
    acceleration in y.

    params
    ===
    x: list like monotonic array.

    y: list like monotonic array.

    best_mid: int, the best, in theory, midpoint of the logistic function for the
    data.

    normalised: bool, if input data is normalised.
        Default: False.

    return
    ===
    np array: [maxima, minima, d_mid, k, auc], round up to 3 digits.
    """
    if not normalised:
        # normalise y
        norm_y = y / y.max()
        # normalise x
        norm_x = x / x.max()
    else:
        norm_y = y
        norm_x = x

    import seaborn as sns
    import matplotlib.pyplot as plt
    # fit normalised data to logistic func
    params, covs = curve_fit(
        f=step_function,
        xdata=norm_x,
        ydata=norm_y,
        p0=[0.8, 1], # give initial guess to make convergence faster
    )

    pred = step_function(x, *params)
    assert 0, "not implemented"
    # solve first derivative to find max & min
    _, np_sec = get_logistic_derivative(*params)
    fs_devs = np_sec(norm_x)
    # highest acceleration
    maxima = x[np.argmax(sec_devs)]
    # highest decceleration
    minima = x[np.argmin(sec_devs)]

    # get distance in cm between fitted midpoint to best midpoint
    d_mid = params[-1] * x.max() - best_mid

    return np.append(params, [maxima, minima, d_mid]).round(3)
