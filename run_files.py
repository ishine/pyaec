import numpy as np
import soundfile as sf
import argparse
import os

from frequency_domain_adaptive_filters.fdkf_sr import fdkf_sr


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    Code src: https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / (np.abs(R)+1e-15), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau

def main(list_file, output_dir, align):
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

    if align:
        tau = gcc_phat(y[:sr * 10], x[:sr * 10], fs=sr, interp=1)
        tau = max(0, int((tau - 0.001) * sr))
        x = np.concatenate([np.zeros(tau), x], axis=0)[:y.shape[-1]]

    print("processing frequency domain kalman filters with 4 convolutive taps")

    e = fdkf_sr(x, y, frame_length=256, window_length=1024, tap_num=4, alpha_q=0.99, alpha_s=0.99)
    # e = fdkf_sr(x, y, frame_length=256, window_length=1024, tap_num=4)
    e = np.clip(e,-1,1)
    sf.write(os.path.join(output_dir, os.path.basename(mic_path).replace(".wav", "_aec_alpha0.99.wav")), e, sr, subtype='PCM_16')
    e = fdkf_sr(x, y, frame_length=256, window_length=1024, tap_num=4, alpha_q=0.95, alpha_s=0.95)
    # e = fdkf_sr(x, y, frame_length=256, window_length=1024, tap_num=4)
    e = np.clip(e,-1,1)
    sf.write(os.path.join(output_dir, os.path.basename(mic_path).replace(".wav", "_aec_alpha0.95.wav")), e, sr, subtype='PCM_16')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--list_file", type=str, required=True, help="a file that contains input file list with the type (mic.wav ref.wav) ")
  parser.add_argument("--output_dir", type=str, required=True, help="output directory")
  parser.add_argument('-a', "--align", action='store_true', help='whether to align x and y')
  args = parser.parse_args()
  main(args.list_file, args.output_dir, args.align)
  