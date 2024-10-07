"""
This module contains math related helper functions to process data.
"""
import multiprocessing as mp
import psutil

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.interpolate import interp1d, PchipInterpolator
import sympy as sp

# initiate random number generator
rng = np.random.default_rng()

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

def _batch_sampling(rng, batch_size, array, size, axis):

    return np.array(
        [rng.choice(array, size=size, axis=axis)
         for _ in range(batch_size)]
    ).astype(int)


def random_sampling(array, size, axis=0, repeats=10000, n_processes=None):
    # TODO for vr session level chance, i.e., randomly choose size from all
    # number of wrong licks within a session, then use that size as the size
    # here to create random samples 

    # define number of processes
    n_processes = n_processes or mp.cpu_count() - 2

    # get batch size in each process
    batch_size = repeats // n_processes
    # create a pool of processes
    pool = mp.Pool(processes=n_processes)
    # apply sampling to each item in repeats
    results = pool.starmap(
        _batch_sampling,
        [(rng, batch_size, array, size, axis)] * n_processes,
    )

    # stop adding task to pool
    pool.close()
    # wait till all tasks in pool completed
    pool.join()

    # concatenate batches of results
    samples = np.concatenate(results)

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
        if x.max() == 0:
            if not isinstance(x, pd.Series):
                x = pd.Series(x)
                norm_x = x.value_counts(normalize=True).values
        else:
            norm_x = x / x.max()

        if y.max() == 0:
            if not isinstance(y, pd.Series):
                y = pd.Series(y)
                norm_y = y.value_counts(normalize=True).values
        else:
            norm_y = y / y.max()

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
        except (IndexError, KeyError):
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


def confidence_interval(array, axis=0, samples=10000, size=25):
    """
    Compute the 95% confidence interval of some data in an array.

    Parameters
    ==========
    array : np.array
        The data, can have any number of dimensions.

    axis : int, optional
        The axis in which to compute CIs. This axis is collapsed in the output. Default:
        0.

    samples : int, optional
        The number of samples to bootstrap. Default: 10000.

    size : int, optional
        The size of each boostrapped sample. Default: 25.

    """
    samps = random_sampling(array, size, axis, samples)

    medians = np.median(samps, axis=-1)
    results = np.percentile(medians, [2.5, 97.5], axis=0)

    return results[1, ...] - results[0, ...]


# Function to process data in chunks
def interpolate_data(timestamps, trial_counts, values, new_timestamps,
                          new_trial_counts, chunk_size=100000):
    interpolated_values = np.empty(new_timestamps.shape)

    num_chunks = len(new_timestamps) // chunk_size + (1 if len(new_timestamps) %
                                                      chunk_size != 0 else 0)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(new_timestamps))

        new_points_chunk = np.array([new_timestamps[start:end], new_trial_counts[start:end]]).T

        # Create a meshgrid for interpolation
        timestamp_grid, trial_count_grid = np.meshgrid(timestamps, trial_counts)

        # Flatten the meshgrid and combine with values for griddata input
        points = np.array([timestamp_grid.flatten(), trial_count_grid.flatten()]).T
        values_grid = np.tile(values, (len(trial_counts), 1)).flatten()

        # Interpolate values at new points chunk
        interpolated_values_chunk = griddata(points, values_grid, new_points_chunk, method='linear')

        # Store the chunk of interpolated values
        interpolated_values[start:end] = interpolated_values_chunk

    return interpolated_values


# Function to initialize multiprocessing
def interpolate_2d(x0, y0, values, x1, y1):
    # TODO: jun 4th this func does NOT work!!!
    """
    Interpolate values from x0 y0 to x1 y1.
    """
    # define number of processes
    n_processes = n_processes or mp.cpu_count() - 2

    # get memory info of the system
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total
    available_memory = memory_info.available

    # reserve some for other stuff
    reserve = total_memory * 0.3
    memory_to_use = available_memory - reserve

    # get size in byte of data
    x0_size = x0[0].dtype.itemsize * array.size
    y0_size = y0[0].dtype.itemsize * array.size
    values_size = values[0].dtype.itemsize * array.size
    data_size = values_size + y0_size + x0_size

    # get batch size in each process
    batch_size = memory_to_use // data_size
    n_batch = int(np.ceil(len(new_timestamps) / batch_size))
    nn_batch = len(new_timestamps) // batch_size\
            + (1 if len(new_timestamps) % batch_size != 0 else 0)

    assert 0
    # create a pool of processes
    pool = mp.Pool(processes=n_processes)

    # get batch data
    batch_data = np.array([x1[start:end], y1[start:end]]).T

    # apply sampling to each item in repeats
    results = [
        pool.apply_async(
            _batch_interpolation,
            args=(i*batch_size, min((i+1)*batch_size, len(x1)), x0, y0,
                  values, x1, y1)) for i in range(n_batch)
    ]
    #results = pool.starmap(
    #    _batch_interpolation,
    #    [(i*batch_size, min((i+1)*batch_size, len(x1)), x0, y0,
    #          values, x1, y1)] * n_processes,
    #)

    pool.close()
    pool.join()

    #interpolated = np.concatenate([result.get() for result in results])
    interpolated = np.concatenate(results)

    return interpolated


# Function to process a single chunk
def _batch_interpolation(start, end, x0, y0, values, x1, y1):
    # TODO: jun 4th this func does NOT work!!!
    # Create a meshgrid for interpolation
    x0_grid, y0_grid = np.meshgrid(x0, y0)
    x1_grid, y1_grid = np.meshgrid(x1, y1)

    new_points_chunk = np.array([x1[start:end], y1[start:end]]).T

    # Flatten the meshgrid and combine with values for griddata input
    points0 = np.array([x0_grid.flatten(), y0_grid.flatten()]).T
    values_grid = np.tile(values, (len(y0), 1)).flatten()

    # Interpolate values
    interpolated = griddata(
        points=points,
        values=values_grid,
        xi=points1,
        method='nearest',
    )

    return interpolated


def interpolate_1d(x, y, xnew, kind='nearest', dtype=float):
    if y.ndim == 1:
        if kind == 'cubic':
            f = PchipInterpolator(
                x=x,
                y=y,
            )
        else:
            # check if x is n-d array
            f = interp1d(
                x=x,
                y=y,
                kind=kind,
                fill_value='extrapolate',
            )
        interpolated = f(xnew)
    else:
        interpolated = np.zeros((xnew.shape[0], y.shape[1]))
        for i in range(y.shape[1]):
            if kind == 'cubic':
                f = PchipInterpolator(
                    x=x,
                    y=y[:, i],
                )
            else:
                # check if x is n-d array
                f = interp1d(
                    x=x,
                    y=y[:, i],
                    kind=kind,
                    fill_value='extrapolate',
                )
            interpolated[:, i] = f(xnew)

    return interpolated.astype(dtype)
