import numpy as np


def colorbar(zmin, zmax, n_ticks, title):
    ticktext = ["{:.2E}".format(val)
                for val in 10**np.linspace(np.log10(zmin), np.log10(zmax),
                                           n_ticks)]
    answer = dict(
        title=title,
        tickmode="array",
        tickvals=np.linspace(np.log10(zmin), np.log10(zmax), n_ticks),
        ticktext=ticktext
    )
    return answer

