import numpy as np


def locallyWeightedLogLikelihood(weight_func, weight_func_scale, logpdf, x,
                                 params_x, data):
    answer = 0.0
    for i in range(len(data)):
        xi = data[i, 0]
        data_i = data[i, 1]
        weight = weight_func(x-xi, scale=weight_func_scale)
        logpdf_val = logpdf(value=data_i, params=params_x)
        answer += weight * logpdf_val
    return answer


def lwll_statistic(data, test_xs, fit_func, weight_func, weight_func_scale,
                   logpdf, x_dist_thr):
    # data[i, 0] == x_i
    # data[i, 1] == theta_i
    answer = 0.0
    for test_x in test_xs:
        indices = np.nonzero(np.abs(test_x-data[:, 0]) < x_dist_thr)[0]
        close_data = data[indices, 1]
        fitted_kappa, fitted_theta0, _ = fit_func(close_data)
        print(f"x; {test_x}, theta: {fitted_theta0}, kappa: {fitted_kappa}, "
              f"nNeighbors: {len(close_data)}")
        params_x = (fitted_theta0, fitted_kappa)
        answer += locallyWeightedLogLikelihood(
            weight_func=weight_func, weight_func_scale=weight_func_scale,
            logpdf=logpdf, x=test_x, params_x=params_x, data=data)
    return answer


def kenneth_statistic(x, theta):
    indices = np.argsort(x)
    sorted_theta = theta[indices]
    sorted_theta_diff = np.diff(sorted_theta)
    stat = np.sum(np.cos(sorted_theta_diff))
    return stat


def kenneth_statistic_modified(x, theta):
    indices = np.argsort(x)
    sorted_theta = theta[indices]
    distance = 0
    for i in range(len(sorted_theta)-1):
        value = 1-np.abs((np.exp(1j*sorted_theta[i+1]) +
                          np.exp(1j*sorted_theta[i]))/2)
        distance += value
        breakpoint()
    return distance
