import numpy as np

def saf_kalman(mic, spk, frame_size):
    out_len = min(len(mic), len(spk))
    out = np.zeros(out_len)
    out_num = out_len // frame_size
    st = init(frame_size)
    
    for i in range(out_num):
        mic_frame = mic[(i - 1) * frame_size : i * frame_size]
        spk_frame = spk[(i - 1) * frame_size : i * frame_size]
        st, out_frame = saf_process(st, mic_frame, spk_frame)
        out[i * frame_size : (i + 1) * frame_size] = out_frame
        
    return out


def init(frame_size):
    st = {}
    # static para
    st['frame_len'] = frame_size
    st['K'] = frame_size * 2
    st['half_bin'] = st['K'] // 2 + 1
    st['win_len'] = st['K'] * 2
    st['notch_radius'] = 0.982
    st['notch_mem'] = np.zeros(2)
    st['memX'] = 0
    st['memD'] = 0
    st['memE'] = 0
    st['preemph'] = 0.98
    
    # lp win 
    win_st = np.load('win_para_update.npy', allow_pickle=True).item()
    st['win_global'] = win_st['win']
    # subband para
    st['ana_win_echo'] = np.zeros(st['win_len'])
    st['ana_win_far'] = np.zeros(st['win_len'])
    st['sys_win'] = np.zeros(st['win_len'])
    st['tap'] = 15
    st['subband_in'] = np.zeros((st['half_bin'], st['tap']))
    st['subband_adf'] = np.zeros((st['half_bin'], st['tap']))
    
    # kalman para
    st['Ryu'] = np.ones((st['half_bin'], st['tap'], st['tap'])) * 5
    st['w_cov'] = np.ones(st['half_bin']) * 0.1
    st['v_cov'] = np.ones(st['half_bin']) * 0.001
    st['gain'] = np.zeros((st['half_bin'], st['tap']))
    
    # nlp 
    st['Eh'] = np.zeros(st['half_bin'])
    st['Yh'] = np.zeros(st['half_bin'])
    st['est_ps'] = np.zeros(st['half_bin'])
    st['spec_ave'] = 0.01
    st['Pey'] = 0
    st['Pyy'] = 0
    st['beta0'] = 0.016
    st['beta_max'] = st['beta0'] / 4
    st['min_leak'] = 0.005
    st['echo_noise_ps'] = 0
    st['adapt_cnt'] = 0
    st['res_old_ps'] = 0
    st['suppress_gain'] = 10
    st['wiener_gain'] = np.zeros(st['half_bin'])
    st['gain_floor'] = np.ones(st['half_bin']) * 0.01
    
    return st


def filter_dc_notch16(in_signal, radius, length, mem):
    out = np.zeros(length)
    den2 = radius * radius + 0.7 * (1 - radius) * (1 - radius)
    for ii in range(length):
        vin = in_signal[ii]
        vout = mem[0] + vin
        mem[0] = mem[1] + 2 * (-vin + radius * vout)
        mem[1] = vin - (den2 * vout)
        out[ii] = radius * vout
    
    return out, mem


def saf_process(st, mic_frame, spk_frame):
    N = st['frame_len']
    K = st['K']
    mic_in, st['notch_mem'] = filter_dc_notch16(mic_frame, st['notch_radius'], N, st['notch_mem'])

    st['ana_win_echo'] = np.concatenate((st['ana_win_echo'][N:], mic_in))
    ana_win_echo_windowed = st['win_global'] * st['ana_win_echo']
    ana_wined_echo = ana_win_echo_windowed[:K] + ana_win_echo_windowed[K:2 * K]
    fft_out_echo = np.fft.fft(ana_wined_echo)

    st['ana_win_far'] = np.concatenate((st['ana_win_far'][N:], spk_frame))
    ana_win_far_windowed = st['win_global'] * st['ana_win_far']
    ana_wined_far = ana_win_far_windowed[:K] + ana_win_far_windowed[K:2 * K]
    fft_out_far = np.fft.fft(ana_wined_far, K)

    st['subband_in'] = np.column_stack((fft_out_far[:st['half_bin']], st['subband_in'][:,:-1]))
    subband_adf_out = np.sum(st['subband_adf'] * st['subband_in'], axis=1)
    subband_adf_err = fft_out_echo[:st['half_bin']] - subband_adf_out
    
    # kalman update
    for j in range(st['half_bin']):
        # update sigmal v
        st['v_cov'][j] = 0.99 * st['v_cov'][j] + 0.01 * (subband_adf_err[j] * subband_adf_err[j])
        
        Rmu = st['Ryu'][j,:,:] + np.eye(st['tap']) * st['w_cov'][j]
        Re = np.real(np.dot(st['subband_in'][j,:], np.dot(Rmu, st['subband_in'][j,:]))) + st['v_cov'][j]
        gain = np.dot(Rmu, st['subband_in'][j,:]) / (Re + 1e-10)
        phi = gain * subband_adf_err[j]
        st['subband_adf'][j,:] = st['subband_adf'][j,:] + phi
        st['Ryu'][j,:,:] = (np.eye(st['tap']) - np.outer(gain, st['subband_in'][j,:])) * Rmu
        
        # update sigmal w
        st['w_cov'][j] = 0.99 * st['w_cov'][j] + 0.01 * (np.sqrt(np.dot(phi, phi.T)) / st['tap'])
    
    # compose subband
    if 1:
        # nlp
        st, nlp_out = nlpProcess(st, subband_adf_err, subband_adf_out)
        ifft_in = np.concatenate((nlp_out, np.flipud(np.conj(nlp_out[1:-1]))))
    else:
        ifft_in = np.concatenate((subband_adf_err, np.flipud(np.conj(subband_adf_err[1:-1]))))
    
    fft_out = np.fft.ifft(ifft_in)
    win_in = np.concatenate((fft_out, fft_out))
    comp_out = win_in * st['win_global']
    st['sys_win'] = st['sys_win'] + comp_out
    out = st['sys_win'][:N]
    st['sys_win'] = np.concatenate((st['sys_win'][N:], np.zeros(N)))
    st['adapt_cnt'] = st['adapt_cnt'] + 1
    
    return st, out


def nlpProcess(st, error, est_echo):
    st['est_ps'] = np.abs(est_echo) ** 2
    res_ps = np.abs(error) ** 2

    Eh_curr = res_ps - st['Eh']
    Yh_curr = st['est_ps'] - st['Yh']
    Pey = np.sum(Eh_curr * Yh_curr)
    Pyy = np.sum(Yh_curr * Yh_curr)
    st['Eh'] = (1 - st['spec_ave']) * st['Eh'] + st['spec_ave'] * res_ps
    st['Yh'] = (1 - st['spec_ave']) * st['Yh'] + st['spec_ave'] * st['est_ps']
    Syy = np.sum(st['est_ps'])
    See = np.sum(res_ps)
    Pyy = np.sqrt(Pyy)
    Pey = Pey / (Pyy + 1e-10)
    tmp32 = st['beta0'] * Syy / See
    alpha = min(tmp32, st['beta_max'])
    st['Pyy'] = (1 - alpha) * st['Pyy'] + alpha * Pyy
    st['Pey'] = (1 - alpha) * st['Pey'] + alpha * Pey
    st['Pyy'] = max(st['Pyy'], 1)
    st['Pey'] = max(st['Pey'], st['Pyy'] * st['min_leak'])
    st['Pey'] = min(st['Pey'], st['Pyy'])
    leak = st['Pey'] / st['Pyy']
    if leak > 0.5:
        leak = 1
    residual_ps = leak * st['est_ps'] * st['suppress_gain']
    
    if st['adapt_cnt'] == 0:
        st['echo_noise_ps'] = residual_ps
    else:
        st['echo_noise_ps'] = max(0.85 * st['echo_noise_ps'], residual_ps)
    st['echo_noise_ps'] = max(st['echo_noise_ps'], 1e-10)
    postser = res_ps / st['echo_noise_ps'] - 1
    postser = np.minimum(postser, 100)
    if st['adapt_cnt'] == 0:
        st['res_old_ps'] = res_ps
    prioriser = 0.5 * np.maximum(0, postser) + 0.5 * (st['res_old_ps'] / st['echo_noise_ps'])
    prioriser = np.minimum(prioriser, 100)
    st['wiener_gain'] = prioriser / (prioriser + 1)
    st['wiener_gain'] = np.maximum(st['wiener_gain'], st['gain_floor'])
    st['res_old_ps'] = 0.8 * st['res_old_ps'] + 0.2 * st['wiener_gain'] * res_ps
    nlp_out = st['wiener_gain'] * error
    
    return st, nlp_out


# Example usage
mic = np.random.randn(1000)  # Example input signals
spk = np.random.randn(1000)
frame_size = 256
output = saf_kalman(mic, spk, frame_size)
print(output)
