import numpy as np
import soundfile as sf
import argparse
import os

from frequency_domain_adaptive_filters.fdkf_sr import fdkf_sr


def main(list_file, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  os.system("rm -rf {}/*".format(output_dir))
  with open(list_file, "r") as f:
    lines = f.readlines()
  for line in lines:
    mic_path, ref_path = line.strip().split()
    print(mic_path, ref_path)
    x, sr = sf.read(ref_path)
    assert sr == 16000
    y, sr = sf.read(mic_path)
    assert sr == 16000

    print("processing frequency domain kalman filters with 4 convolutive taps")

    e = fdkf_sr(x, y, frame_length=256, window_length=1024, tap_num=4)
    e = np.clip(e,-1,1)
    sf.write('samples/fdkf_sr.wav', e, sr, subtype='PCM_16')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--list_file", type=str, required=True, help="a file that contains input file list with the type (mic.wav ref.wav) ")
  parser.add_argument("--output_dir", type=str, required=True, help="output directory")
  args = parser.parse_args()
  main(args.list_file, args.output_dir)
  