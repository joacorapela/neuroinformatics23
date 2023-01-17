import sys
import argparse
import pickle
import numpy as np
import one.api


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=str, help="experiment to analyze",
                       default="ebe2efe3-e8a1-451a-8947-76ef42427cc9")
    parser.add_argument("--probe_id", type=str, help="id of the probe to analyze",
                       default="probe00")
    parser.add_argument("--elapsed_start", type=float,
                       help="elapsed time (in secs) between trials start times and the go cue",
                       default=0.1)
    parser.add_argument("--elapsed_end", type=float,
                       help="elapsed time (in secs) between the go cue and trials end times",
                       default=1.0)
    parser.add_argument("--save_filename_pattern", type=str,
                       help="filename where to save the epoched spikes",
                       default="../../results/epoched_spike_times_eID{:s}_probeID{:s}.pickle")
    args = parser.parse_args()

    eID = args.experiment_id
    probe_id = args.probe_id
    elapsed_start = args.elapsed_start
    elapsed_end = args.elapsed_end
    save_filename_pattern = args.save_filename_pattern

    aOne = one.api.ONE(base_url='https://openalyx.internationalbrainlab.org',
              password='international', silent=True)
    spikes = aOne.load_object(eID, 'spikes', f'alf/{probe_id}/pykilosort')
    trials = aOne.load_object(eID, 'trials')
    goCue_times = trials['goCue_times']
    goCue_times = goCue_times[np.logical_not(np.isnan(goCue_times))]

    clusters_id = np.unique(spikes.clusters)
    n_neurons = len(clusters_id)
    n_trials = len(goCue_times)
    epoched_spikes_times = [None for r in range(n_trials)]
    for r in range(n_trials):
        print(f"Processing trial {r} ({n_trials-1})")
        epoched_spikes_times[r] = [None for n in range(n_neurons)]
        for n in range(n_neurons):
            neuron_spikes_times = spikes.times[spikes.clusters==n]
            start_indices = np.searchsorted(neuron_spikes_times, goCue_times-elapsed_start)
            end_indices = np.searchsorted(neuron_spikes_times, goCue_times+elapsed_end)
            epoched_spikes_times[r][n] = neuron_spikes_times[start_indices[r]:end_indices[r]]-goCue_times[r]

    save_filename = save_filename_pattern.format(eID, probe_id)
    results = {"spikes_times": epoched_spikes_times, "behavioral_data": trials}
    with open(save_filename, "wb") as f:
        pickle.dump(results, f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
