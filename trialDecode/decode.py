import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import logging
import os
import tqdm


def extract_dat_channel(dat_filename: str, n_chs_total: int, serial_ch_num: int, datatype='int16')\
        -> np.ndarray:
    """
    
    :param dat_filename: path to binary file.
    :param n_chs_total: total number of channels in the binary file
    :param serial_ch_num: This is the number of the channel that contains the serial stream. INDEXING
    STARTS AT 0!
    :return: 
    """
    if not os.path.exists(dat_filename):
        logging.error('No dat file at {}'.format(dat_filename))
        raise FileNotFoundError('No file at {}')
    logging.info('Opening dat {}'.format(dat_filename))
    dat = np.memmap(dat_filename, dtype=datatype, mode='r')
    dat.shape = (-1, n_chs_total)
    logging.info('Reading serial stream at index {}'.format(serial_ch_num))
    ss = dat[:, serial_ch_num]  # unfortunately we're slicing this on the slow axis.
    ss_array = np.zeros(ss.shape, ss.dtype)
    blockends = np.linspace(0, len(ss_array), 21, dtype=np.int)
    for i in tqdm.tqdm(range(1, len(blockends)), desc='Loading stream'):
        st = blockends[i-1]
        nd = blockends[i]
        ss_array[st:nd] = ss[st:nd]
    return ss_array


def parse_serial_stream(stream, fs=25000):
    """
    Finds serial numbers if they exist in a stream.
    Uses a more intelligent method to describe the
    threshold value within the stream.
    (fit with 1 or 2 gaussians. If fit w/ 1 is better, no opens. Else, threshold is between the two means)

    :param stream: stream to find threshold crossings.
    :return: list of tuples [(starttimes, decoded number)]
    """
    logging.info("Parsing serial stream...")

    # first determine if our stream is fit better by 1 or two gaussian distributions:
    params = _gaussian_model_comparison(stream)
    if len(params) > 2:
        mu1, mu2, s1, s2 = params
        # if it is fit by 2 gaussians, next check to see if the gaussians are really distinct:
        if np.abs(mu1 - mu2) > s1 * 6:
            threshold = np.min([mu1, mu2]) + np.abs(mu1-mu2)/2
            threshold = threshold.astype(stream.dtype)
        else:
            threshold = None
    else:  # if the stream is fit better by one gaussian, then we don't have any fv opens.
        threshold = None

    if threshold is not None:  #if we have set a threshold above, find the finalvalve opens.
        logging.debug('using threshold {}'.format(threshold))
        trial_times = _parse_serial(stream, fs, threshold=threshold, )
    else:
        trial_times = []
    logging.info('Complete. {} trial starts found.'.format(len(trial_times)))

    return trial_times


def _parse_serial(serial_stream, fs=25000., word_len=2, baudrate=300, threshold=None):
    """

    :param serial_stream: array containing serial stream.
    :param fs: sampling frequency of serial stream
    :param word_len: number of bytes per word (ie 16 bit int is 2 bytes)
    :param baudrate: baudrate of serial transmission.
    :param threshold: threshold to use to extract high vs low states. If None, use stream.max / 2
    :return: list of tuples [(time, decoded_number)]
    """

    # NOTE: The bit spacing here is only approximately right due to a problem with arduino timing that makes the endbit
    # between bytes longer than they should be (20% for 300 baud). Ideally we should parse bytes individually, but this
    # works (at least with this baudrate). You would have to be 50% off before this should cause a problem because it is
    # reading the bits from the center of where they should be.

    if serial_stream.ndim == 2:
        try:
            shortdim = serial_stream.shape.index(1)
            if shortdim == 1:
                serial_stream = serial_stream[:, 0]
            else:
                serial_stream = serial_stream[0, :]
        except ValueError:
            raise ValueError('_parse_serial input must be a 1d array or 2d with dimension of length 1.')

    start_bit = 1
    end_bit = 1
    bit_len = float(fs)/baudrate

    if threshold is None:
        # this is a really crappy way of finding the threshold, to be sure.
        threshold = np.max(serial_stream)/2

    log_stream = (serial_stream > threshold)
    edges = np.convolve(log_stream, [1,-1])
    ups = np.where(edges == 1)[0]
    downs = np.where(edges == -1)[0]

    #check to see if baudrate/fs combo is reasonable by looking for bits of the length specified by them in the stream.
    diff = (downs-ups) / bit_len
    cl = 0
    for i in range(1,4):  # look for bit widths of 1,2,3 consecutive.
        diff -= 1.
        g = diff > -.1
        l = diff < .1
        cl += np.sum(g*l)
    if cl < 4:
        print ('WARNING: bits do not appear at timings consistent with selected \nsampling frequency and baud rate ' \
              'combination: (fs: %i, baud: %i). This warning may be erroneous for short recordings.' % (fs, baudrate))

    start_id = downs-ups > int(bit_len * word_len * (8 + start_bit + end_bit))
    start_samples = downs[start_id]

    # Arduino software serial can start initially low before the first transmission, so the start detector above will fail.
    # This means that the initial start bit is masked, as the signal is already low. We're going to try to find the
    # end bit of this ridiculous first word, and work backward to find where this masked start bit occured.
    if not log_stream[0]:
        try:
            for i in range(len(ups)-1):
                if (ups[i+1] - ups[i]) > (bit_len * word_len * (8 + start_bit + end_bit)):

                    firstword_end = ups[i]
                    break
        except IndexError:
            raise ValueError('Serial parsing error: stream starts low, but cannot find end of first word.')
        bts = word_len * (8+start_bit+end_bit) - 1 # end bit of the last byte IS the up, so we shouldn't count it.
        firstword_start = firstword_end - (bts*bit_len)
        firstword_start = np.int(firstword_start)
        start_samples = np.concatenate(([firstword_start], start_samples))

    word_bits = bit_len * word_len * (8+start_bit+end_bit)
    bit_sample_spacing = np.linspace(bit_len/2, word_bits - bit_len/2, word_len * 10)
    bit_sample_spacing = bit_sample_spacing.round().astype(int)

    _bytes = np.empty((word_len, 8), dtype=bool)

    trial_times = []
    for start in start_samples[:-1]:
        bit_samples = bit_sample_spacing + start
        bits = log_stream[bit_samples]
        for i in range(word_len):
            # This strips serial protocol bits assuming that a serial byte looks like: [startbit, 0, 1, ..., 7, endbit]
            # as per normal serial com protocol. In other words: returned _bytes are 8 bit payloads.
            _bytes[i,:] = bits[i*10+1:i*10+9]
        number = _bytes_to_int(_bytes)
        trial_times.append((start, number))
    return trial_times


def _bytes_to_int(bytes, endianness='little'):
    """

    :param bytes: np boolean array with shape == (number_bytes, 8)
    :param endianness of the encoded bytes
    :return:
    """
    bits = bytes.ravel()
    if endianness == 'big':
        bits = np.flipud(bits)
    num = 0
    for e, bit in enumerate(bits):
        num += bit * 2 ** e
    return num





def _gaussian_model_comparison(stream):
    """
    TTL values are modeled as value plus gaussian noise. Find if the stream data is explained better by
    one or two gaussians. We're determining model fit by fitting using MLE and using BIC to compare model
    fits (while penalizing for number of parameters, althought this isn't a big issue.

    :param stream:
    :return:
    """
    N = len(stream)
    maxN = 30000
    if N > maxN:
        dec = int(np.ceil(N/maxN))  # we're guaranteed to get at least 15000 samples here which should be ok.
        assert dec > 0
    else:
        dec = 1

    X = stream[::dec].astype(np.float64)
    N_X = len(X)
    mu_bound1 = np.min(X)
    mu_bound2 = np.max(X)
    mu_st1 = max((mu_bound1+1., 100.))
    mu_st2 = min((mu_bound2-1., 70000.))
    fit1 = _fit_gaussian(X,  μ_bounds=(mu_bound1,mu_bound2))
    fit2 = _fit_sum_of_2gaussians(X,  μ_bounds=(mu_bound1,mu_bound2))
    for f in (fit1, fit2):
        assert f.success

    ic1 = _bic(fit1.fun, 2., N_X)
    ic2 = _bic(fit2.fun, 4., N_X)

    if ic1 < ic2:  # lower BIC wins
        result = fit1.x
        logging.info('1 gaussian fit is better. {}'. format(result))
    else:
        result = fit2.x
        logging.info('2 gaussian fit is better. {}'.format(result))
    return result


def _fit_gaussian(x,
                  x0=(15000., 5.),
                  μ_bounds=(0,30000.),
                  σ_bounds=(.1,8000.)):
    """

    :param x: array of observations to fit (should be float64)
    :param x0: starts (μ, σ)
    :param μ_bounds: bounds for the mean (loc) parameters
    :param σ_bounds: bounds for std parameter (scale)
    :return:
    """
    assert len(x) <= 30000, 'Too many values will make this over/underflow.'
    bounds = (μ_bounds, σ_bounds)

    def nll(args):
        μ, σ = args
        return -norm.logpdf(x, loc=μ, scale=σ).sum()

    return minimize(nll, x0, bounds=bounds)

def _fit_sum_of_2gaussians(x,
                           x0=(400., 15000., 1., 10.),
                           μ_bounds=(0., 70000.),
                           σ_bounds=(.01, 3000.)):
    """

    :param x: array of observations to fit (should be float64)
    :param x0: starts (μ1, μ2, σ1, σ2)
    :param μ_bounds: bounds for the mean (loc) parameters. Will be used for both gaussians.
    :param σ_bounds: bounds for std parameter (scale). Will be used for both gaussians.
    :return:
    """

    assert len(x) <= 30000, 'Too many values will make this over/underflow.'
    bounds = (μ_bounds, μ_bounds, σ_bounds, σ_bounds)

    def nll(args):  #objective function to optimize.
        μ1, μ2, σ1, σ2 = args
        f1 = norm.logpdf(x, loc=μ1, scale=σ1)
        f2 = norm.logpdf(x, loc=μ2, scale=σ2)
        ss = np.logaddexp(f1, f2)  # only way to add these tiny numbers on a computer. read the fn docs.
        return -ss.sum() + np.log(0.5)  # log identity: log(a/2) = log(a) + log(1/2)

    return minimize(nll, x0, bounds=bounds)

def _bic(nlogL, k, n):
    """
    Bayesian Information Criterion
    nlogL is the _negative_ log likelihood of the model.
    k is number of parameters in model.
    n is the number of observations that were used to fit the model. (ie the length of the array)
    """
    return 2. * nlogL + k * np.log(n)


def split_to_runs(trialstarts: list) -> list:
    """
    Breaks trial numbers into putative runs. Within runs, the following condition is always met:
    trial[i+1] > trial[i]. If this condition is not met, then we are starting a new run.

    This returns a structure: [[r1_t1, ..., r1_tN], [r2_t1, ..., r2_tN]]

    :param meta_file: meta tb.File object.
    :return: list of runs, each of which contains a list of trial starts (time, trial_number)
    """

    ts_by_run = []
    ts_run = []
    curr_trial = -1
    for st in trialstarts:
        t, tnum = st
        if tnum > 0:
            if curr_trial > tnum or curr_trial == tnum == 1:
                # we've had restart? I don't like that this is not explicit. maybe we can assume restarts start at "1"
                ts_by_run.append(ts_run)
                ts_run = []
                curr_trial = -1
            elif curr_trial == st[1]:
                raise ValueError('duplicate trial numbers in sequence.')
            ts_run.append(st)
            curr_trial = tnum
    ts_by_run.append(ts_run)
    return ts_by_run