from . import script
import argparse

desc = """
A program to trial numbers from voyeur + ephys.\n\n

Currently, supports decoding trial numbers from a single ephys stream file, but can match these trial numbers\n
to multiple Voyeur Data files.
"""

parser = argparse.ArgumentParser(prog='trialDecode', description=desc)
parser.add_argument('stream_filename', help='Path to dat or openephys files.', type=str)
parser.add_argument('voyeur_filenames', nargs='+', type=str, help='Paths to voyeur files. (Can be multiple)')
parser.add_argument('serial_channel', help='Index of channel where serial stream is recorded.', type=int)
parser.add_argument('number_channels', help='Number of channels recorded in dat file.', type=int)
parser.add_argument('--dtype', default='int16', type=str, nargs=1, help='Datatype (default int16)')
parser.add_argument('-r', '--sample_rate', help='Sample rate in hz fo recording.', default=25000, type=int)
parser.add_argument('-s', '--save_to', help='Save to path.', default='', type=str)
parser.add_argument('--skip', help='Number of samples to skip at beginning of serial stream.', type=int, default=0)
parser.add_argument('--truncate', help='Number of samples to ignore at the end of serial stream.', type=int, default=-1)


def main():
    args = parser.parse_args()
    script.run(
        args.stream_filename,
        args.voyeur_filenames,
        args.serial_channel,
        args.number_channels,
        args.dtype,
        args.sample_rate,
        args.save_to,
        args.skip,
        args.truncate
    )

if __name__ == '__main__':
    main()