"""
This module contains math related helper functions to process data.
"""
import multiprocessing as mp
import psutil

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress, norm, anderson, bootstrap
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import periodogram, welch
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


def _batch_sampling(batch_size, a, size, axis):
    """
    Randomly sample batch.

    params
    ===
    batch_size: int, number of items in a batch.

    a: array like or int, sampling source.

    size: int, number of samples to take.

    axis: int, axis along which the selection is done.
    
    return
    ===
    samples: nd array, randomly selected samples, shape (batch_size, size).

    """
    sample = np.array(
        [rng.choice(a, size=size, replace=True, axis=axis)
        for _ in range(batch_size)]
    ).astype(int)

    # sort sample
    sample.sort()

    return sample


def random_sampling(a, size, axis=0, repeats=10000, n_processes=None):
    """
    Repeatedly randomly sample from a given array, or np.arange(a).

    params
    ===
    a: array like or int, sampling source.

    size: int, number of samples to take.

    axis: int, axis along which the selection is done.
    
    repeats: int, number of repeat.

    n_processes: int, number of cpu cores to use.

    return
    ===
    samples: nd array, randomly selected samples, shape (repeats, size).
    """

    # define number of processes
    n_processes = n_processes or mp.cpu_count() - 2

    # get batch size in each process
    base_batch = repeats // n_processes
    remainder = repeats % n_processes
    # create tasks
    tasks = []
    if remainder > 0:
        for i in range(n_processes):
            # Distribute remainder: first 'remainder' processes get an extra sample.
            batch_size = base_batch + (1 if i < remainder else 0)
            # Use i as a seed to differentiate RNG instances
            tasks.append((batch_size, a, size, axis))

    # create a pool of processes
    pool = mp.Pool(processes=n_processes)
    # apply sampling to each item in repeats
    results = pool.starmap(
        _batch_sampling,
        tasks,
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
    np array: [k, x_0, maxima, minima, d_mid], round up to 3 digits.
    """
    if x.size < 2 or y.size < 2:
        logging.info("> Not enough data points to fit logistics.")

        return np.full(5, np.nan)

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
        #logging.info(f"\n> Fitted logistic function is not differentiable, most likely",
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

def log_likelihood(data, x):
    """
    Get log likelihood of the observation x in given data.

    NOTE: likelihood is NOT probability!
    Likelihood reflects relative plausibility the parameters given x, i.e., 
    L(theta|x_0, x_1, ..., x_n); while probability is the probability of
    observing x given the parameters, i.e., P(x|theta).

    params
    ===
    data: np array, use to infer std and mean of data.

    x: float or int, observation.
    """
    # get standard deviation
    sigma = np.std(data)
    # get mean
    mu = np.mean(data)
    # get log likelihood L from log pdf
    log_L = norm.logpdf(
        x=x,
        loc=mu,
        scale=sigma,
    )

    return log_L


def group_and_aggregate(df, group_key, how, columns=None):
    if columns:
        grouped = df.groupby(group_key)[columns]
    else:
        grouped = df.groupby(group_key)

    # find the needed attribute of groupby
    agg_method = getattr(grouped, how)

    return agg_method()


def kendalls_w(ranks) -> float:
    """
    Compute Kendall’s W (coefficient of concordance) for *m* raters
    ranking *n* items.
    params
    ===
    ranks: pandas df or np array, shape (n_items, m_raters), each column is a
        COMPLETE ranking of the same n_items, from 0…n-1 (or 1…n, doesn’t matter so
        long as it’s consecutive integers).
    return
    ===
    W: float, between [0, 1].
    """
    if isinstance(ranks, pd.DataFrame):
        R = ranks.values.astype(float)
    elif isinstance(ranks, np.array):
        assert 0
        R = ranks

    n, m = R.shape
    # sum of ranks for each item across all raters
    sum_r = np.sum(R, axis=1)
    # the “mean total rank” if rater‐agreement were perfect
    R_bar = m * (n - 1) / 2.0
    # S = ∑ (R_i – R̄)²
    S = np.sum((sum_r - R_bar)**2)
    # denominator: m² (n³ – n) / 12
    W = (12 * S) / (m ** 2 * (n ** 3 - n))

    return W


def kendalls_w_test(ranks):
    """
    Returns (W, chi2_stat, df, p_value) testing H0: W=0 versus W>0
    by the chi‐square approximation.
    """
    W = kendalls_w(rank_df)
    n, m = rank_df.shape
    chi2_stat = m * (n - 1) * W
    df = n - 1
    pval = 1 - chi2.cdf(chi2_stat, df)
    return W, chi2_stat, df, pval


def estimate_power_spectrum(x, fs=1.0, axis=-1, scaling="density",
                            use_welch=True):
    """
    Estimate power spectrum or spectral density using periodogram.

    params
    ===
    x: array like, time series or similar sequence of measurement values, in
        unit of spike/second (Hz).

    fs: float, sampling frequency of x, in unit of sample / cycle. this "cycle"
        could be spike/second (Hz, in temporal domain), or sample/centimetre (in
        spatial domain).


    axis: int, axis along which the periodogram is computed.
        Default: -1

    scaling: str, scaling method of power spectrum.
        if "density", psx unit would be (sample/cycle)^2

    use_welch: bool, use welch or periodogram.
        Default: True, it gives smoother, low variance estimate of the psd.

    return
    ===
    sample_freqs: array of sample frequencies.

    psx: power spectrum or spectral density (in unit of Hz^2/cm).
    """
    if use_welch:
        sample_freqs, psx = welch(
            x=x,
            fs=fs,
            window="hann",
            scaling=scaling,
            nperseg=256,
            return_onesided=True,
            detrend="linear",
            axis=axis,
            average="median",
        )
    else:
        sample_freqs, psx = periodogram(
            x=x,
            fs=fs,
            window="hann",
            scaling=scaling,
            detrend="linear",
            axis=axis,
        )

    return sample_freqs, psx


def normality_check(df):
    """
    Normality check using Anderson-Darling test, robust against big sample size.

    params
    ===
    df: pd dataframe, time x unit.
    """
    normality_check = []
    for i in range(len(df.columns)):
        results = anderson(df[df.columns[i]], dist="norm")
        normality_check.append((results.statistic, results.fit_result.success))

    normality = pd.DataFrame(
        normality_check,
        columns=["A2", "success"],
        index=df.columns,
    )
    logging.info("normality check:\n", normality)

    return normality


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


def bootstrap_ci(arr, axis, ci=0.95, target_stat="mean", method="basic",
                 n_resamples=1000):
    """
    Compute bootstrap CI on the mean of arr along axis=0.

    params
    ===
    arr : np.ndarray, shape = (n_trials, n_timepoints)

    returns
    ===
    (low, high), each shape (n_timepoints,)
    """
    # `statistic` must consume arr and return mean over axis=0
    if target_stat == "mean":
        stat = np.mean
    if target_stat == "median":
        stat = np.median

    res = bootstrap(
        (arr,),
        statistic=stat,
        confidence_level=ci,
        batch=100,
        n_resamples=n_resamples,
        method=method,# "BCa" for more accurate
        vectorized=True,
        axis=axis,
        random_state=rng,
    )
    return res.confidence_interval.low, res.confidence_interval.high


def _batch_permute(batch_size, a, axis):
    """
    Randomly sample batch.

    params
    ===
    batch_size: int, number of items in a batch.

    a: array like or int, sampling source.

    size: int, number of samples to take.

    axis: int, axis along which the selection is done.
    
    return
    ===
    samples: nd array, randomly selected samples, shape (batch_size, size).

    """
    rng = np.random.default_rng()

    sample = np.array(
        [rng.permutation(a, axis=axis) for _ in range(batch_size)]
    )

    return sample


def random_permutation(a, axis=0, repeats=10000, n_processes=None):
    """
    Repeatedly randomly sample from a given array, or np.arange(a).

    params
    ===
    a: array like or int, sampling source.

    size: int, number of samples to take.

    axis: int, axis along which the selection is done.
    
    repeats: int, number of repeat.

    n_processes: int, number of cpu cores to use.

    return
    ===
    samples: nd array, randomly selected samples, shape (repeats, size).
    """

    # define number of processes
    n_processes = n_processes or mp.cpu_count() - 2

    # get batch size in each process
    base_batch = repeats // n_processes
    remainder = repeats % n_processes
    # create tasks
    tasks = []
    if remainder > 0:
        for i in range(n_processes):
            # Distribute remainder: first 'remainder' processes get an extra sample.
            batch_size = base_batch + (1 if i < remainder else 0)
            # Use i as a seed to differentiate RNG instances
            tasks.append((batch_size, a, axis))

    # create a pool of processes
    pool = mp.Pool(processes=n_processes)
    # apply sampling to each item in repeats
    results = pool.starmap(
        _batch_permute,
        tasks,
    )

    # stop adding task to pool
    pool.close()
    # wait till all tasks in pool completed
    pool.join()

    # concatenate batches of results
    samples = np.concatenate(results)

    return samples
