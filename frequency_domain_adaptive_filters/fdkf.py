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

""" Frequency Domain Kalman Filter """

import numpy as np
from librosa.core import stft, istft

def fdkf(ref, mic, frame_length, window_length, filter_length, beta=0.95, sgm2u=1e-2, sgm2v=1e-6):
  x = ref
  d = mic
  n_freq = window_length//2+1
  X = stft(x, n_fft=window_length, win_length=window_length, hop_length=frame_length, center=False)
  D = stft(d, n_fft=window_length, win_length=window_length, hop_length=frame_length, center=False)

  Q = sgm2u
  R = np.full(n_freq,sgm2v)
  H = np.zeros(n_freq,dtype=np.complex)
  P = np.full(n_freq,sgm2u)

  num_block = X.shape[-1]
  assert num_block == (x.shape[-1] - window_length) // frame_length + 1

  for n in range(num_block):
    X_n = X[:,n]
    D_n = D[:,n]
    Y_n = H.conj()*X_n
    E_n = D_n-Y_n

    R = beta*R + (1.0 - beta)*(np.abs(E_n)**2)
    P_n = P + Q*(np.abs(H))
    K = P_n*X_n.conj()/(X_n*P_n*X_n.conj()+R)
    P = (1.0 - K*X_n)*P_n 

    H = H + K*E_n
    h = ifft(H)
    h[M:] = 0
    H = fft(h)

    e[n*M:(n+1)*M] = e_n
  
  return e