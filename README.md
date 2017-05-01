#trialDecode

This a small package to decode binary trial number information from binary streams 
and sync with Voyeur trials.

In the end, you will get a JSON with the following structure:

       [
        {START_SAMPLE=1000,  # for first trial, start time in ephys samples.
         Voyeur_trial_param1='value'
       }
        {START_SAMPLE=2000,  # for second trial...
         Voyeur_trial_param1='value'
       }
      ]
       
##Installation:
Installation is easy, but requires the following packages. I recommend using Conda python distribution.
All required packages are found on Conda.

1. Python >3.5
2. Numpy
3. Scipy
4. PyTables (`conda install pytables`)
5. tqdm (must be installed: `conda install tqdm`)

To install, navigate to this folder in your console, and run:

`python setup.py install`

Test the installation by running:

`trialDecode -h`

If you are having problems with the installation, it may be that your path is not pointing to the conda
Python interpreter. Double check with `which python` and update your path if needed!


## Use
Use of the script is simple. It has several required positional arguments:


1. path to binary
2. path(s) to Voyeur H5 files
3. serial stream channel _(actually the index of the channel in the binary file, with indexing starting at 0)_
4. total number channels contained binary


It also has several optional arguments.:
1. -r (--sample_rate): sample rate in hz
2. -s (--save_to): specify a save file name
3. --skip N: allows you to skip the first N samples in the binary stream
4. --truncate N: allows you to skip the last N samples in the binary stream

###Typical usage:
for one voyeur file and a dat file with 86 channels where the serial stream is at index 86:
```
trialDecode a_02_data_g1_t0.nidq.bin a_02_D2017_3_16T11_32_36_beh.h5 84 86
```

for two voyeur files and a dat file with 86 channels where the serial stream is at index 86:
```
trialDecode a_02_data_g1_t0.nidq.bin voyeur_file1.h5 voyeur_file2.h5 84 86
```
