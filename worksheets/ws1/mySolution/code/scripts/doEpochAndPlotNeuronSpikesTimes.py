import sys
import argparse
import pickle
import numpy as np
import plotly.graph_objs as go
import one.api


def getSpikesTimesPlotOneNeuron(spikes_times,
                                sorting_times,
                                neuron_index, title,
                                trials_ids,
                                behavioral_times_col=[],
                                behavioral_times_labels=[],
                                marked_events_times=None,
                                marked_events_colors=None,
                                marked_events_markers=None,
                                align_event=None,
                                marked_size=10, spikes_symbol="line-ns-open",
                                trials_colors=None, default_trial_color="black",
                                xlabel="Time (sec)", ylabel="Trial",
                                event_line_color="rgba(0, 0, 255, 0.2)",
                                event_line_width=5, spikes_marker_size=9):
    if sorting_times is not None:
        argsort = np.argsort(sorting_times)
        spikes_times = [spikes_times[r] for r in argsort]
        sorting_times = [sorting_times[r] for r in argsort]
        trials_ids = [trials_ids[r] for r in argsort]
        for i, behavioral_times in enumerate(behavioral_times_col):
            sorted_behavioral_times = [behavioral_times[r] for r in argsort]
            behavioral_times_col[i] = sorted_behavioral_times
    n_trials = len(trials_ids)
    fig = go.Figure()
    for r in range(n_trials):
        spikes_times_trial_neuron = spikes_times[r][neuron_index]
        # workaround because if a trial contains only one spike spikes_times[n]
        # does not respond to the len function
        if spikes_times_trial_neuron.size == 1:
            spikes_times_trial_neuron = [spikes_times_trial_neuron]
        if trials_colors is not None:
            spikes_color = trials_colors[r]
        else:
            spikes_color = default_trial_color
        trial_label = "{:02d}".format(trials_ids[r])
        trace = go.Scatter(
            x=spikes_times_trial_neuron,
            y=r*np.ones(len(spikes_times_trial_neuron)),
            mode="markers",
            marker=dict(size=spikes_marker_size, color=spikes_color,
                        symbol=spikes_symbol),
            name="trial {:s}".format(trial_label),
            legendgroup=f"trial{trial_label}",
            showlegend=False,
            text=[trial_label]*len(spikes_times_trial_neuron),
            hovertemplate="Time %{x}<br>" + "Trial %{text}",
        )
        fig.add_trace(trace)
        if marked_events_times is not None:
            marked_events_times_centered = marked_events_times[r]-align_event[r]
            n_marked_events = len(marked_events_times[r])
            for i in range(n_marked_events):
                trace_marker = go.Scatter(x=[marked_events_times_centered[i]],
                                          y=[r],
                                          marker=dict(color=marked_events_colors[r][i],
                                                      symbol=marked_events_markers[r][i],
                                                      size=marked_size),
                                          name="trial {:s}".format(trial_label),
                                          text=[trial_label],
                                          hovertemplate="Time %{x}<br>" + "Trial %{text}",
                                          mode="markers",
                                          legendgroup=f"trial{trial_label}",
                                          showlegend=False)
                fig.add_trace(trace_marker)
    for i, behavioral_times in enumerate(behavioral_times_col):
        trace = go.Scatter(x=behavioral_times, y=np.arange(n_trials),
                           name=behavioral_times_labels[i])
        fig.add_trace(trace)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title=title)
    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )
    return fig

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_event_name", type=str,
                        help="behavioral event name to use to epoch",
                        default="goCue_times")
    parser.add_argument("--sorting_event_name", type=str,
                        help="behavioral event name to use to sort trials",
                        default=None)
    parser.add_argument("--behavioral_events_names", type=str,
                        help="behavioral events names to plot",
                        default="[goCue_times,stimOn_times,response_times,feedback_times]")
    parser.add_argument("--colors_event_name", type=str,
                        help="events names used to color spikes", default="choice")
    parser.add_argument("--experiment_id", type=str, help="experiment to analyze",
                       default="ebe2efe3-e8a1-451a-8947-76ef42427cc9")
    parser.add_argument("--probe_id", type=str, help="id of the probe to analyze",
                       default="probe00")
    parser.add_argument("--cluster_id", type=int, help="cluster_id to analyze",
                       default=41)
    parser.add_argument("--elapsed_start", type=float,
                       help="elapsed time (in secs) between trials start times and the go cue",
                       default=0.1)
    parser.add_argument("--elapsed_end", type=float,
                       help="elapsed time (in secs) between the go cue and trials end times",
                       default=1.0)
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/spikes_times_clusterID{:d}_epochedBy{:s}_sortedBy{:s}_colors{:s}.{:s}")
    args = parser.parse_args()

    epoch_event_name = args.epoch_event_name
    sorting_event_name = args.sorting_event_name
    behavioral_events_names = args.behavioral_events_names[1:-1].split(",")
    colors_event_name = args.colors_event_name
    eID = args.experiment_id
    probe_id = args.probe_id
    cluster_id = args.cluster_id
    elapsed_start = args.elapsed_start
    elapsed_end = args.elapsed_end
    fig_filename_pattern = args.fig_filename_pattern

    aOne = one.api.ONE(base_url='https://openalyx.internationalbrainlab.org',
              password='international', silent=True)
    spikes = aOne.load_object(eID, 'spikes', f'alf/{probe_id}/pykilosort')
    trials = aOne.load_object(eID, 'trials')

    epoch_times = trials[epoch_event_name]
    keep_trial = np.logical_not(np.isnan(epoch_times))
    epoch_times = epoch_times[keep_trial]
    if sorting_event_name is not None:
        sorting_times = trials[sorting_event_name]
        sorting_times = sorting_times[keep_trial]
        sorting_times -= epoch_times
    else:
        sorting_times = None
    behavioral_times_col = [None for i in range(len(behavioral_events_names))]
    for i, behavioral_event_name in enumerate(behavioral_events_names):
        behavioral_times_col[i] = trials[behavioral_event_name]
        behavioral_times_col[i] = behavioral_times_col[i][keep_trial]
        behavioral_times_col[i] -= epoch_times

    colors_event = trials[colors_event_name]
    colors_event = colors_event[keep_trial]

    n_trials = len(epoch_times)
    neuron_spikes_times = spikes.times[spikes.clusters==cluster_id]
    start_indices = np.searchsorted(neuron_spikes_times, epoch_times-elapsed_start)
    end_indices = np.searchsorted(neuron_spikes_times, epoch_times+elapsed_end)
    epoched_spikes_times = \
        [[neuron_spikes_times[start_indices[r]:end_indices[r]]-epoch_times[r]]
         for r in range(n_trials)]

    trials_colors = ["red" if an_event==1 else "blue" for an_event in colors_event]

    title=f"Neuron: {cluster_id}, Epoched by: {epoch_event_name}, Sorted by: {sorting_event_name}, Spike colors by: {colors_event_name}"
    trials_ids = np.arange(n_trials)
    fig = getSpikesTimesPlotOneNeuron(
        spikes_times=epoched_spikes_times,
        sorting_times=sorting_times,
        neuron_index=0,
        behavioral_times_col=behavioral_times_col,
        behavioral_times_labels=behavioral_events_names,
        title=title,
        trials_colors=trials_colors,
        trials_ids=trials_ids,
    )
    fig.update_xaxes(range=[-elapsed_start, elapsed_end])
    fig.write_image(fig_filename_pattern.format(cluster_id, epoch_event_name,
                                                sorting_event_name,
                                                colors_event_name, "png"))
    fig.write_html(fig_filename_pattern.format(cluster_id, epoch_event_name,
                                               sorting_event_name,
                                               colors_event_name, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
