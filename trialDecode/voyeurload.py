import tables as tb
import datetime
import numpy as np
import logging


def load_trials(voyeur_filepaths):

    # First we want to sort our files so that our trials are guaranteed to be in order:
    times = [_get_date_from_voyeur_name(x) for x in voyeur_filepaths]
    _, sorted_paths = zip(*sorted(zip(times, voyeur_filepaths)))

    # Now open ordered files and append Trials arrays together.
    all_trials = []
    for p in sorted_paths:
        logging.info('loading trials from {}'.format(p))
        with tb.open_file(p, 'r') as f:  # type: tb.File
            trials = f.get_node('/Trials').read()
            all_trials.append(trials)
    return all_trials


def align_trials(voyeur_trials, stream_trial_starts) -> list:
    """
    Loads ephys trial starts and behavior trial starts. Aligns the two. Returns a list of trial start times with
    corresponding lines from voyeur behavior files.

    Runs are not enumerated in the returned structure. In other words, runs are handled correctly, but
    trials are returned in the sequence they occurred regardless of run.

    :param meta_file: open meta tb.File object
    :return:  [(t1_start_time, t1_table_row), ... (tN_start_time, tN_table_row)]
    """


    # if not len(voyeur_trials) == len(stream_trial_starts):
    #     erst = "Different number of runs is detected in \
    #         Voyeur nodes ({}) and recorded trial starts ({})".format(len(voyeur_trials), len(stream_trial_starts))
    #     raise ValueError(erst)

    all_trials = []
    for v_trials, e_trials in zip(voyeur_trials, stream_trial_starts):
        v_trial_nums = v_trials['trialNumber']
        e_trial_nums = np.array([x[1] for x in e_trials])
        intersecting_trial_numbers = np.intersect1d(v_trial_nums, e_trial_nums)
        a = np.setdiff1d(v_trial_nums, e_trial_nums)
        b = np.setdiff1d(e_trial_nums, v_trial_nums)

        if len(a):
            logging.warning("Voyeur trials {} were not found in recording.".format(a))
        if len(b):
            logging.warning("Trial numbers: {} found in recording but not in voyeur file.".format(b))
        for tn in intersecting_trial_numbers:
            _v_i = np.where(v_trial_nums == tn)[0]
            assert (len(_v_i) == 1)
            v_i = _v_i[0]
            _e_i = np.where(e_trial_nums == tn)[0]
            assert (len(_e_i) == 1)
            e_i = _e_i[0]
            tr = v_trials[v_i]
            start_time, _ = e_trials[e_i]
            all_trials.append((start_time, tr))
    return all_trials


def _get_date_from_voyeur_name(nodename):
    """
    Parses node names with the following format:
        "***MOUSEDATA***Dyyyy_mm_ddThh_mm_ss"

    :param nodename: 
    :return: 
    """

    _, dt = nodename.split('D')
    d_str, t_str = dt.split('T')
    y, m, d = [int(x) for x in d_str.split('_')]
    h, mn, s = [int(x) for x in t_str.split('_')[:3]]  # beh would be the 4th element
    return datetime.datetime(y, m, d, h, mn, s)



