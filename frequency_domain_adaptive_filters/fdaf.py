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
from librosa.core import stft

def fdaf(x, d, M, mu=0.05, beta=0.9):
  X = stft(x,n_fft=M,win_length=M,center=True)
  H = np.zeros(M+1,dtype=np.complex)
  norm = np.full(M+1,1e-8)

  window =  np.hanning(M)
  x_old = np.zeros(M)
  d_old = np.zeros(M)

  num_block = min(len(x),len(d)) // M
  e = np.zeros(num_block*M)

  for n in range(num_block):
    x_n = np.concatenate([x_old,x[n*M:(n+1)*M]])
    d_n = np.concatenate([d_old,d[n*M:(n+1)*M]])
    x_old = x[n*M:(n+1)*M]
    d_old = d[n*M:(n+1)*M]

    X_n = fft(x_n)
    D_n = fft(d_n)
    Y_n = H.conj()*X_n
    E_n = D_n-Y_n
    e_n = ifft(E_n)[M:]
    e[n*M:(n+1)*M] = e_n * window

    norm = beta*norm + (1-beta)*np.abs(X_n)**2
    # G = mu*E_n/(norm+1e-3)
    H = H + mu * X_n*E_n.conj()/(norm+1e-5)

    # h = ifft(H)
    # h[M:] = 0
    # H = fft(h)

  return e