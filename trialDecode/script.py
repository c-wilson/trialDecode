from . import decode, voyeurload
import json
import os
import logging
import numpy as np

LOG_FORMATTER = logging.Formatter("%(asctime)s %(levelname)-7.7s:  %(message)s")
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(LOG_FORMATTER)
LOGGER.addHandler(console_handler)



def run(stream_path, voyeur_paths, stream_ch, nchs,
        dtype='int16', fs=25000, savefilename=None, skip=0, truncate=-1):
    """
    Synchronizes trial information from binary stream and voyeur files.
    
    :param stream_path: path to binary file
    :param voyeur_paths: paths to voyeur files, list.
    :param stream_ch: channel index for serial stream.
    :param nchs: Number of channels total in the binary file 
    :param dtype: datatype for binary file (default 'int16')
    :param fs: sample rate of binary file in Hz (default 25000)
    :param savefilename: where to save json
    :param skip: number of samples to skip from beginning of serial stream (for debugging)
    :param truncate: number of samples to skip from end of serial stream (for debugging)
    :return: 
    """
    log_fn = "trialDecode.log"
    log_file_handler = logging.FileHandler(log_fn)
    log_file_handler.setFormatter(LOG_FORMATTER)
    LOGGER.addHandler(log_file_handler)

    logging.info('{} total trials detected in serial stream.'.format(len('hello')))
    if not os.path.exists(stream_path):
        raise FileNotFoundError(' Stream file missing: {}'.format(stream_path))
    for p in voyeur_paths:
        if not os.path.exists(p):
            raise FileNotFoundError('Voyeur file missing: {}'.format(p))

    serial_stream = decode.extract_dat_channel(stream_path, nchs, stream_ch, dtype)
    if truncate < 0:
        truncate = None  # array[st:None] is equivalent to array[st:].
    trial_numbers = decode.parse_serial_stream(serial_stream[skip:truncate], fs)
    logging.info('{} total trials detected in serial stream.'.format(len(trial_numbers)))
    trial_numbers_by_run = decode.split_to_runs(trial_numbers)
    logging.info('{} runs detected in serial stream.'.format(len(trial_numbers_by_run)))
    if len(trial_numbers_by_run) != len(voyeur_paths):
        logging.warning('Number of runs detected in serial stream is different from number of voyeur files specified.')
    all_voyeur_trials = voyeurload.load_trials(voyeur_paths)
    aligned = voyeurload.align_trials(all_voyeur_trials, trial_numbers_by_run)
    save_list = _to_dict(aligned)

    if not savefilename:
        st, _ = os.path.splitext(stream_path)
        savefilename = st + '_trials.json'
    elif not savefilename.endswith('.json'):
        savefilename = savefilename + '.json'

    logging.info('Saving to {}'.format(savefilename))
    with open(savefilename, 'w') as f:
        json.dump(save_list, f, indent="\t", )
    LOGGER.removeHandler(log_file_handler)


def _to_dict(trials):
    """
    Returns a list of dictionaries from a trial structure.
    
    :param trials: Trial structure [(start_time1, trial_row_structured_array1), ...] 
    :param savefilename: where to save the json.
    :return: list of dictionaries to save as json.
    """

    to_save = []
    for t in trials:
        trial_start, trial_data = t
        trial_dict = {'START_SAMPLE': int(trial_start)}
        for fieldname in trial_data.dtype.names:
            f = trial_data[fieldname]
            if type(f) == np.ndarray:
                f2 = f.tolist()
            elif f.dtype == np.floating:
                f2 = float(f)
            elif f.dtype == np.integer:
                f2 = int(f)
            elif f.dtype == np.bytes_:
                f2 = f.decode()
            trial_dict[fieldname] = f2

        to_save.append(trial_dict)
    return to_save

