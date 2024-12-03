# Copyright 2020 ewan xu<ewan_xu@outlook.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

""" Frequency Domain Adaptive Filter """

import numpy as np
from librosa.core import stft, istft

def fdaf(ref, mic, frame_length, window_length, tap_num, mu=0.05, beta=0.9):
  x = ref
  d = mic
  n_freq = window_length//2+1
  X = stft(x, n_fft=window_length, win_length=window_length, hop_length=frame_length, center=False)
  D = stft(d, n_fft=window_length, win_length=window_length, hop_length=frame_length, center=False)
  E = np.zeros(X.shape, dtype=np.complex)
  H = np.zeros((n_freq, tap_num), dtype=np.complex)
  norm = np.zeros((n_freq), dtype=np.complex)
  num_block = X.shape[-1]
  assert num_block == (x.shape[-1] - window_length) // frame_length + 1
  X_n = np.zeros((n_freq,tap_num), dtype=np.complex)

  for n in range(num_block):
    X_n[:,1:] = X_n[:,:-1]
    X_n[:,0] = X[:,n]
    D_n = D[:,n]
    Y_n = (H.conj()*X_n).sum(-1)
    E_n = D_n-Y_n
    E[:,n] = E_n

    if np.abs(X_n).mean() < 1e-5:
      continue

    norm = beta*norm + (1-beta)*np.abs(X_n[:,0])**2
    H = H + mu * X_n*E_n.conj()[:,None]/(norm[:,None]+1e-5)

  e = istft(E, win_length=window_length, hop_length=frame_length, center=False)
  return e