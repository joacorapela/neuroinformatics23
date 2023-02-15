
import numpy as np
import scipy.signal
import plotly.graph_objects as go

import one.api
import brainbox.io.one
import brainbox.io.spikeglx


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


pid = "38124fca-a0ac-4b58-8e8b-84a2357850e6"
channel_nro = 250
sxx_win_len_sec = 1200
segment_len = 2**14
n_ticks = 6
colorscale = "viridis"
x_label = "Time (sec)"
y_label = "Frequency (Hz)"
ylim = [0, 160]

aOne = one.api.ONE(base_url="https://openalyx.internationalbrainlab.org",
                   password="international", silent=True)
sr = brainbox.io.spikeglx.Streamer(pid=pid, one=aOne, remove_cached=False,
                                   typ="lf")
sxx_win_len_samples = int(sxx_win_len_sec * sr.fs)
# extract channel location acronyms for hover
eID, probe_label = aOne.pid2eid(pid=pid)
els = brainbox.io.one.load_channel_locations(eID, one=aOne)
channel_locs_acronyms = els[probe_label]["acronym"]

for s0 in range(0, sr.ns, sxx_win_len_samples):
    sf = min(s0 + sxx_win_len_samples, sr.ns)
    selected_samples = slice(s0, sf)
    channel_lfp = sr[selected_samples, channel_nro]*1000
    f, t, Sxx = scipy.signal.spectrogram(x=channel_lfp, fs=sr.fs,
                                         nperseg=segment_len,
                                         scaling="spectrum")
    zmin = np.min(Sxx)
    zmax = np.max(Sxx)
    start_time = s0/sr.fs

    # let's plot now
    title = (f"Sxx (mv^2): channel {channel_nro}, "
             f"region {channel_locs_acronyms[channel_nro]}, "
             f"segment_len {segment_len}")
    fig = go.Figure()
    trace = go.Heatmap(x=t+start_time, y=f, z=np.log10(Sxx),
                       colorscale=colorscale, zmin=np.log10(zmin),
                       zmax=np.log10(zmax),
                       colorbar=colorbar(zmin=zmin, zmax=zmax, n_ticks=n_ticks,
                                         title="Power (mv^2)"))
    fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label, range=ylim)
    fig.update_layout(title=title)
    fig.show()
