
import sys
import numpy as np
import scipy.signal
import plotly.graph_objects as go

import one.api
import brainbox.io.one
import brainbox.io.spikeglx

import plotUtils

pid = "38124fca-a0ac-4b58-8e8b-84a2357850e6"
channel_nro = 250
segment_len = 2**14
n_ticks = 6
colorscale = "viridis"
x_label = "Time (sec)"
y_label = "Frequency (Hz)"
xlim = [4500.0,5500.0]
ylim = [0, 160]

band = "lf"

aOne = one.api.ONE(base_url="https://openalyx.internationalbrainlab.org",
   password="international", silent=True)
sr = brainbox.io.spikeglx.Streamer(pid=pid, one=aOne, remove_cached=False,
   typ=band)
# extract channel location acronyms for hover
eID, probe_label = aOne.pid2eid(pid=pid)
els = brainbox.io.one.load_channel_locations(eID, one=aOne)
channel_locs_acronyms = els[probe_label]["acronym"]

# Important: remove sync channel from raw data, and transpose
channel_lfp = sr[0:sr.ns, channel_nro]*1000
n_samples = len(channel_lfp)
print(f"Data has {n_samples} samples")

f, t, Sxx = scipy.signal.spectrogram(x=channel_lfp, fs=sr.fs,
 nperseg=segment_len,
 scaling="spectrum")
zmin = np.min(Sxx)
zmax = np.max(Sxx)

# let's plot now
# hovertext = []
# for yi, yy in enumerate(f):
# hovertext.append([])
# for xi, xx in enumerate(t):
# hovertext[-1].append(f"time: {xx}<br />frequency: {yy}<br />Sxx: {Sxx[yi][xi]}<br />")

title = f"Sxx (mv^2): channel {channel_nro}, region {channel_locs_acronyms[channel_nro]}, segment_len {segment_len}"
fig = go.Figure()
trace = go.Heatmap(x=t, y=f, z=np.log10(Sxx),
   colorscale=colorscale, zmin=np.log10(zmin),
   zmax=np.log10(zmax),
   colorbar=plotUtils.colorbar(
   zmin=zmin, zmax=zmax, n_ticks=n_ticks,
   title="Power (mv^2)"),
#hoverinfo="text", text=hovertext,
   )
fig.add_trace(trace)
fig.update_xaxes(title_text=x_label, range=xlim)
fig.update_yaxes(title_text=y_label, range=ylim)
fig.update_layout(title=title)

fig.show()
