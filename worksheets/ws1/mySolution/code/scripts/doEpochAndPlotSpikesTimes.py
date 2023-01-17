import sys
import argparse
import pickle
import numpy as np
import plotly.graph_objs as go
import one.api


def getSpikesTimesPlotOneNeuron(spikes_times, neuron_index, title,
                                trials_ids,
                                marked_events_times=None,
                                marked_events_colors=None,
                                marked_events_markers=None,
                                align_event=None,
                                marked_size=10, spikes_symbol="line-ns-open",
                                trials_colors=None, default_trial_color="black",
                                xlabel="Time (sec)", ylabel="Trial",
                                event_line_color="rgba(0, 0, 255, 0.2)",
                                event_line_width=5, spikes_marker_size=9):
    n_trials = len(spikes_times)
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
    parser.add_argument("--experiment_id", type=str, help="experiment to analyze",
                       default="ebe2efe3-e8a1-451a-8947-76ef42427cc9")
    parser.add_argument("--probe_id", type=str, help="id of the probe to analyze",
                       default="probe00")
    parser.add_argument("--cluster_id", type=int, help="cluster_id to analyze",
                       default=41)
    parser.add_argument("--save_filename_pattern", type=str,
                        help="filename where to save the epoched spikes",
                        default="../../results/epoched_spike_times_eID{:s}_probeID{:s}.pickle")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/spikes_times_clusterID{:d}.{:s}")
    args = parser.parse_args()

    eID = args.experiment_id
    probe_id = args.probe_id
    cluster_id = args.cluster_id
    save_filename_pattern = args.save_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    save_filename = save_filename_pattern.format(eID, probe_id)
    with open(save_filename, "rb") as f:
        load_res = pickle.load(f)

    spikes_times = load_res["spikes_times"]
    behavioral_data = load_res["behavioral_data"]

    choices = behavioral_data["choice"]
    trials_colors = ["red" if choice==1 else "blue" for choice in choices]

    n_trials = len(spikes_times)
    trials_ids = np.arange(n_trials)
    fig = getSpikesTimesPlotOneNeuron(
        spikes_times=spikes_times, neuron_index=cluster_id,
        title=f"Neuron: {cluster_id}",
        trials_colors=trials_colors,
        trials_ids=trials_ids,
    )
    fig.write_image(fig_filename_pattern.format(cluster_id, "png"))
    fig.write_html(fig_filename_pattern.format(cluster_id, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
