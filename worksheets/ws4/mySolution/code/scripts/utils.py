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
        params_x = (fitted_theta0, fitted_kappa)
        answer += locallyWeightedLogLikelihood(
            weight_func=weight_func, weight_func_scale=weight_func_scale,
            logpdf=logpdf, x=test_x, params_x=params_x, data=data)
    return answer
