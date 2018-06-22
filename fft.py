import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import scipy.signal as signal

"""plot fft part"""

prime_number = 413
fs = 50e6
fft_size = 4096
v_fs = 1.2
n = 12
v_cm = 0.6
mismatchStd = 0.001
radix = 2
flag = 'bad'
seed = 1
sar_adc = SAR_ADC(v_fs, n, v_cm, mismatchStd, radix, flag, seed)
sar_adc_ideal = SAR_ADC(v_fs=v_fs, n=n, v_cm=v_cm, mismatchStd=0, radix=2, flag=flag, seed=seed)

def getSNR(fft_output, window):
    """function to caculate SNR"""
    max_index = prime_number #the index of max value
    if window == 'hann':
        a_signal = np.array([fft_output[max_index-1:max_index+2]]) #the
        a_noise = np.append(fft_output[3:(max_index-1)], fft_output[(max_index+2):])
    else:
        a_signal = np.array([fft_output[max_index]])
        a_noise = np.append(fft_output[3:(max_index)], fft_output[(max_index+1):])
    p_signal = np.linalg.norm(a_signal) #abs(np.power(a_signal, 2))
    p_noise = np.linalg.norm(a_noise) #abs(np.power(a_noise, 2))
    SNR = 20*np.log10(p_signal/p_noise)
    return SNR

def getENOB(SNR):
    """function to caculate ENOB"""
    ENOB = (SNR-1.7609)/6.0206
    return ENOB


def plot_fft(prime_number, fs, fft_size, sar_adc):
    """function to plot the fft"""
    f_in = fs/fft_size*prime_number #define the input frequency
    x = np.linspace(0, 1, fs+1)
    x = x[:fft_size]
    y_in = sar_adc.amp*np.sin(2*np.pi*f_in*x) + sar_adc.amp #define the input signal
    y_adc_output = [sar_adc.main(y_in_i)[-1] for y_in_i in y_in[:fft_size]] #caculate the SAR ADC output
    y_dac_output = [sar_adc_ideal.DAC(y_i) for y_i in y_adc_output] #caculate the analog signal from ADC output
    y_dac_win = y_dac_output*signal.hann(fft_size+1)[:-1]     #add a hann window function
    y_dac_unwin  = y_dac_output
    
    x_f = np.array(range(0,int(fft_size/2+1)))/fft_size*fs #caculate the frequency value of x axis
    
    y_f_w = np.fft.rfft(y_dac_win, 4096) #caculate the fft given windowed analog output
    y_fn_w = y_f_w/((fft_size/2)*((v_fs-v_cm)/2)) #normalization
    y_p_w = 20*np.log10(y_fn_w)  #caculate the power
    SNR_w = getSNR(y_f_w, 'hann') #caculate the SNR of windowed analog output
    
    y_f_unw = np.fft.rfft(y_dac_unwin, 4096)
    y_fn_unw = y_f_unw/((fft_size/2)*((v_fs-v_cm)/1)) #normalization
    y_p_unw = 20*np.log10(y_fn_unw)
    SNR_unw = getSNR(y_f_unw, None)
    
    print(SNR_w)
    print(SNR_unw)
    print(getENOB(SNR_w))
    print(getENOB(SNR_unw))
    
    #plot part
    plt.subplot(211)
    plt.title('windowed PSD(Hann)')
    plt.plot(x_f[2:], y_p_w[2:],linewidth=0.5)
    plt.ylim((-120, 0))
    plt.text(0, -20,'SNR:%f'%SNR_w,fontsize='x-small')
    plt.text(0,-30,'ENOB:%f'%getENOB(SNR_w),fontsize='x-small')
    plt.grid()
    plt.xlabel('Frequency/Hz')
    plt.ylabel('V/dB')
    
    plt.subplot(212)
    plt.title('unwindowed PSD')
    plt.text(0,-20,'SNR:%f'%SNR_unw,fontsize='x-small')
    plt.text(0,-30,'ENOB:%f'%getENOB(SNR_unw),fontsize='x-small')
    plt.plot(x_f[2:], y_p_unw[2:],linewidth=0.5)
    plt.ylim((-120, 0))
    plt.xlabel('Frequency/Hz')
    plt.ylabel('V/dB')
    plt.grid()
    plt.subplots_adjust(hspace = 1)
    plt.show()

plot_fft(prime_number, fs, fft_size, sar_adc)

