
import numpy as np
import scipy.stats


def get_pvalues_hist(distribution="Normal", mean=0.0, std=1.0, n_samples=10000,
                     n_repeats=1000, popmean=0.0, n_bins=20):
    bins = np.arange(0, 1+1.0/n_bins, 1.0/n_bins)
    p_values = [None] * n_repeats

    for i in range(n_repeats):
        if distribution == "Normal":
            sample = np.random.normal(loc=mean, scale=std, size=n_samples)
        elif distribution == "StdCauchy":
            sample = np.random.normal(size=n_samples)
        elif distribution == "Rademacher":
            uniforms = np.random.uniform(size=n_samples)
            sample = np.where(uniforms < 0.5, 1, -1)
        elif distribution == "VerySkewed":
            uniforms = np.random.uniform(size=n_samples)
            sample = np.where(uniforms < 1e-3, 1, 0)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        _, p_values[i] = scipy.stats.ttest_1samp(sample, popmean=popmean)
    p_values_hist, _ = np.histogram(p_values, bins=bins)
    count_p_values_less0_05 = np.sum(np.array(p_values) < 0.05)

    return p_values_hist, count_p_values_less0_05