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

def fdkf_tencent(ref, mic, frame_length, window_length, tap_num):
  A = 0.9995
  alpha_q = 0.9
  n_taps = tap_num
  x = ref
  y = mic
  n_freq = window_length//2+1
  X = stft(x, n_fft=window_length, win_length=window_length, hop_length=frame_length, center=False).T
  Y = stft(y, n_fft=window_length, win_length=window_length, hop_length=frame_length, center=False).T
  E = np.zeros(X.shape, dtype=np.complex64)


  X_m = np.zeros((n_taps, n_freq), dtype=np.complex64)
  H_prior = np.zeros((n_taps, n_freq), dtype=np.complex64)
  H_post = np.zeros((n_taps, n_freq), dtype=np.complex64)
  # H_prior = np.full((n_freq, 1), 1e-6, dtype=np.complex)
  # H_post = np.full((n_freq, 1), 1e-6, dtype=np.complex)
  # P_post = np.full((n_freq, n_freq), 1e-6, dtype=np.complex)
  P = np.full((n_freq, n_taps, n_taps), 1e-6, dtype=np.float32  )
  P_post = np.full((n_freq, n_taps, n_taps), 1e-6, dtype=np.float32)
  Q = np.full((n_freq, n_taps, n_taps), 1e-6, dtype=np.float32)
  

  num_block = X.shape[0]
  assert num_block == (x.shape[-1] - window_length) // frame_length + 1

  for m in range(num_block):
    X_m = np.roll(X_m, 1, axis=0)
    X_m[0,:] = X[m,:]
    if np.abs(X_m).mean() < 1e-5:
      E[m,k] = Y[m,k]
      continue
    for k in range(n_freq):
      Y_mk = Y[m,k]
      X_mk = X_m[:,k].reshape(-1,1)
      H_prior[:,k] = A * H_post[:,k]
      Q[k,:,:] = alpha_q * Q[k,:,:] + (1 - alpha_q) * (1 - A**2) * H_prior[:,k].reshape(-1,1).dot(H_prior[:,k].conj().reshape(1,-1)).real
      P[k,:,:] = A ** 2 * P_post[k,:,:] + Q[k,:,:]
      E_temp = Y_mk - H_prior[:,k].conj().dot(X_mk)
      K = P[k,:,:].dot(X_mk) / (X_mk.conj().T.dot(P[k,:,:]).dot(X_mk) + np.abs(E_temp) ** 2)
      H_post[:,k] = H_prior[:,k] + (K * E_temp).reshape(-1)
      P_post[k,:,:] = (np.eye(n_taps) - K.dot(X_mk.conj().T)).dot(P[k,:,:])
      E[m,k] = Y_mk - H_post[:,k].conj().dot(X_mk)

  e = istft(E.T, win_length=window_length, hop_length=frame_length, center=False)
  return e