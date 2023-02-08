import sys
import argparse
import pickle
import numpy as np
import plotly.graph_objs as go
import one.api
import brainbox.io.one

def epoch_neuron_spikes_times(neuron_spikes_times, epoch_times,
                              epoch_start_times, epoch_end_times,
                              elapsed_start, elapsed_end):
    epoch_start_indices = np.searchsorted(neuron_spikes_times, epoch_start_times-elapsed_start)
    epoch_end_indices = np.searchsorted(neuron_spikes_times, epoch_end_times+elapsed_end)
    epoch_end_indices[epoch_end_indices==len(neuron_spikes_times)] = len(neuron_spikes_times)-1
    n_trials = len(epoch_start_indices)
    epoched_spikes_times = \
        [[neuron_spikes_times[epoch_start_indices[r]:epoch_end_indices[r]]-epoch_times[r]]
         for r in range(n_trials)]
    return epoched_spikes_times


def getSpikesTimesPlotOneNeuron(spikes_times,
                                sorting_times,
                                neuron_index, title,
                                trials_ids,
                                feedback_types,
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
        feedback_types = [feedback_types[r] for r in argsort]
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
        feedback_type = "{:d}".format(int(feedback_types[r]))
        trace = go.Scatter(
            x=spikes_times_trial_neuron,
            y=r*np.ones(len(spikes_times_trial_neuron)),
            mode="markers",
            marker=dict(size=spikes_marker_size, color=spikes_color,
                        symbol=spikes_symbol),
            name="trial {:s}".format(trial_label),
            legendgroup=f"trial{trial_label}",
            showlegend=False,
            text=[f"Trial {trial_label}<br>Feedback {feedback_type}"]*len(spikes_times_trial_neuron),
            hovertemplate="Time %{x}<br>%{text}",
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
                        default="response_times")
    parser.add_argument("--epoch_start_event_name", type=str,
                        help="behavioral event name to use to start epochs",
                        default="stimOn_times")
    parser.add_argument("--epoch_end_event_name", type=str,
                        help="behavioral event name to use to end epochs",
                        default="stimOff_times")
    parser.add_argument("--sorting_event_name", type=str,
                        help="behavioral event name to use to sort trials",
                        default=None)
    parser.add_argument("--feedback_type_event_name", type=str,
                        help="feedback type event name to use hover",
                        default="feedbackType")
    parser.add_argument("--behavioral_events_names", type=str,
                        help="behavioral events names to plot",
                        default="[stimOn_times,goCue_times,response_times,feedback_times,stimOff_times]")
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
                       default=2.0)
    parser.add_argument("--elapsed_end", type=float,
                       help="elapsed time (in secs) between the stimulus offset time and a trial end time",
                       default=2.0)
    parser.add_argument("--xmax", type=float, help="maximum x-axis value",
                        default=5.0)
    parser.add_argument("--xmin", type=float, help="mininum x-axis value",
                        default=-2.0)
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/spikes_times_eid{:s}_probeid{:s}_clusterid{:d}_epochedBy{:s}_sortedBy{:s}_colors{:s}.{:s}")
    args = parser.parse_args()

    epoch_event_name = args.epoch_event_name
    epoch_start_event_name = args.epoch_start_event_name
    epoch_end_event_name = args.epoch_end_event_name
    sorting_event_name = args.sorting_event_name
    feedback_type_event_name = args.feedback_type_event_name
    behavioral_events_names = args.behavioral_events_names[1:-1].split(",")
    colors_event_name = args.colors_event_name
    eID = args.experiment_id
    probe_id = args.probe_id
    cluster_id = args.cluster_id
    elapsed_start = args.elapsed_start
    elapsed_end = args.elapsed_end
    xmin = args.xmin
    xmax = args.xmax
    fig_filename_pattern = args.fig_filename_pattern

    aOne = one.api.ONE(base_url='https://openalyx.internationalbrainlab.org',
              password='international', silent=True)
    spikes = aOne.load_object(eID, 'spikes', f'alf/{probe_id}/pykilosort')
    trials = aOne.load_object(eID, 'trials')

    clusters = aOne.load_object(eID, "clusters", f"alf/{probe_id}/pykilosort")
    els = brainbox.io.one.load_channel_locations(eID, one=aOne)
    channel_for_cluster_id = clusters.channels[cluster_id]
    region_for_cluster_id = els[probe_id]["acronym"][channel_for_cluster_id]

    epoch_times = trials[epoch_event_name]
    n_trials = len(epoch_times)
    trials_ids = np.arange(n_trials)

    epoch_start_times = trials[epoch_start_event_name]
    epoch_end_times = trials[epoch_end_event_name]
    feedback_types = trials[feedback_type_event_name]
    remove_trial = np.logical_or(np.isnan(epoch_times),
                                 np.logical_or(np.isnan(epoch_start_times),
                                               np.isnan(epoch_end_times)))
    keep_trial = np.logical_not(remove_trial)
    epoch_times = epoch_times[keep_trial]
    epoch_start_times = epoch_start_times[keep_trial]
    epoch_end_times = epoch_end_times[keep_trial]
    feedback_types = feedback_types[keep_trial]
    trials_ids = trials_ids[keep_trial]
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

    neuron_spikes_times = spikes.times[spikes.clusters==cluster_id]
    epoched_spikes_times = epoch_neuron_spikes_times(
        neuron_spikes_times=neuron_spikes_times,
        epoch_times = epoch_times,
        epoch_start_times=epoch_start_times,
        epoch_end_times=epoch_end_times,
        elapsed_start=elapsed_start,
        elapsed_end=elapsed_end)

    trials_colors = ["red" if an_event==1 else "blue" for an_event in colors_event]

    title=f"Neuron: {cluster_id}, Region: {region_for_cluster_id}, Epoched by: {epoch_event_name}, Sorted by: {sorting_event_name}, Spike colors by: {colors_event_name}"
    fig = getSpikesTimesPlotOneNeuron(
        spikes_times=epoched_spikes_times,
        sorting_times=sorting_times,
        neuron_index=0,
        behavioral_times_col=behavioral_times_col,
        behavioral_times_labels=behavioral_events_names,
        title=title,
        trials_colors=trials_colors,
        trials_ids=trials_ids,
        feedback_types=feedback_types,
    )

    if xmin is None:
        xmin = np.min(neuron_spikes_times[epoch_start_indices]-epoch_times)
    if xmax is None:
        xmax = np.max(neuron_spikes_times[epoch_end_indices]-epoch_times)
    fig.update_xaxes(range=[xmin, xmax])

    fig.write_image(fig_filename_pattern.format(eID, probe_id, cluster_id,
                                                epoch_event_name,
                                                sorting_event_name,
                                                colors_event_name, "png"))
    fig.write_html(fig_filename_pattern.format(eID, probe_id, cluster_id,
                                               epoch_event_name,
                                               sorting_event_name,
                                               colors_event_name, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
