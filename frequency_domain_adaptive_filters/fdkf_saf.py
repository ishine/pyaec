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

def fdkf_saf(ref, mic, frame_length, window_length, tap_num, alpha_q=0.95, alpha_s=0.95, A=0.999):
  A = 0.999
  alpha_q = alpha_q
  alpha_s = alpha_s
  n_taps = tap_num
  x = ref
  y = mic
  n_freq = window_length//2+1
  X = stft(x, n_fft=window_length, win_length=window_length, hop_length=frame_length, center=False).T
  Y = stft(y, n_fft=window_length, win_length=window_length, hop_length=frame_length, center=False).T
  E = np.zeros(X.shape, dtype=np.complex64)


  X_m = np.zeros((n_taps, n_freq), dtype=np.complex64)
  H_post = np.zeros((n_taps, n_freq), dtype=np.complex64)
  # H_prior = np.full((n_freq, 1), 1e-6, dtype=np.complex)
  # H_post = np.full((n_freq, 1), 1e-6, dtype=np.complex)
  # P_post = np.full((n_freq, n_freq), 1e-6, dtype=np.complex)
  P_post = np.full((n_freq, n_taps, n_taps), 1e-2, dtype=np.float32)
  Q = np.full((n_freq, n_taps, n_taps), 1e-6, dtype=np.float32)
  S = np.full((n_freq,), 1e-6, dtype=np.float32)
  w_conv = np.full((n_freq), 1e-2, dtype=np.float32)
  

  num_block = X.shape[0]
  assert num_block == (x.shape[-1] - window_length) // frame_length + 1

  for m in range(num_block):
    # X_m = np.roll(X_m, 1, axis=0)
    X_m[1:,:] = X_m[:-1,:]
    X_m[0,:] = X[m,:]
    if np.abs(X_m).mean() < 1e-5:
      E[m,:] = Y[m,:]
      continue
    for k in range(n_freq):
      Y_mk = Y[m,k]
      X_mk = X_m[:,k].reshape(-1,1)
      # H_prior = (A * H_post[:,k]).reshape(-1,1)
      # Q[k,:,:] = alpha_q * Q[k,:,:] + (1 - alpha_q) * (1 - A**2) * H_prior.dot(H_prior.conj().T).real
      # # Q[k,:,:] = H_prior.dot(H_prior.conj().T).real
      # P = A ** 2 * P_post[k,:,:] + (1 - A ** 2) * Q[k,:,:]
      # E_temp = Y_mk - H_prior.conj().T.dot(X_mk)
      # S[k] = alpha_s * S[k] + (1 - alpha_s) * np.abs(E_temp).real ** 2
      # K = P.dot(X_mk) / (X_mk.conj().T.dot(P).dot(X_mk) + S[k])
      # H_post[:,k] = (H_prior + (K * E_temp)).reshape(-1)
      # P_post[k,:,:] = (np.eye(n_taps) - K.dot(X_mk.conj().T)).dot(P)
      # E[m,k] = Y_mk - H_post[:,k].conj().dot(X_mk)

      # H_prior = H_post[:,k].reshape(-1,1)
      # E_temp = Y_mk - H_prior.T.dot(X_mk)
      # S[k] = alpha_s * S[k] + (1 - alpha_s) * (np.abs(E_temp).real ** 2)
      # P =  P_post[k] + 1e-2 * np.sqrt(H_prior.dot(H_prior.conj().T).real)
      # K = P.dot(X_mk.conj()) / (X_mk.T.dot(P).dot(X_mk.conj()) + S[k])
      # P_post[k] = (np.eye(n_taps) - K.dot(X_mk.T)).dot(P)
      # H_post[:,k] = (H_prior + (K * E_temp)).reshape(-1)
      # E[m,k] = Y_mk - H_post[:,k].dot(X_mk)


      H_prior = H_post[:,k].reshape(-1,1)
      E_temp = Y_mk - H_prior.T.dot(X_mk)
      S[k] = alpha_s * S[k] + (1 - alpha_s) * (np.abs(E_temp).real ** 2)
      P =  P_post[k] + np.eye(n_taps) * w_conv[k]
      K = P.dot(X_mk.conj()) / (X_mk.T.dot(P).dot(X_mk.conj()) + S[k])
      d_H = K * E_temp
      H_post[:,k] = (H_prior + d_H).reshape(-1)
      E[m,k] = Y_mk - H_post[:,k].dot(X_mk)
      P_post[k] = (np.eye(n_taps) - K.dot(X_mk.T)).dot(P)
      w_conv[k] = alpha_q * w_conv[k] + (1 - alpha_q) * np.sqrt(np.abs(d_H.T.conj().dot(d_H)).real)/n_taps

      


  e = istft(E.T, win_length=window_length, hop_length=frame_length, center=False)
  return e