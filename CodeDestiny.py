import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt

class SAR_ADC:
    """SAR ADC class with mismatch"""
    
    def __init__(self, v_fs=8, n=3, v_cm=4, mismatchStd=0.0,  radix=2, flag='good', seed=1):
        """initial values"""
        self.v_fs = v_fs #full scale volt of SAR ADC
        self.n = n #number of bits
        self.v_cm = v_cm
        self.amp = np.max([v_fs - v_cm, v_cm-0])
        self.v_ref = v_fs #reference volt
        self.mismatchStd = mismatchStd #mismatch stand
        self.radix = radix #radix of capacitance array
        self.flag = flag #mismatch flag('good' or 'bad')
        self.seed = seed #random seed, used to fix the random values.
        self.sar_logic = {-1:1, 1:0} #SAR logic
        self.capArray = self.synthesizeCapArray() #according to mismatchStd and random seed to generate a capacitance array
    
    def DAC(self, N):
        """function to caculate dac-output by given v_ref, N(a string of N-bit digital input like '1011')"""
        weights = self.capArray[:-1]/np.sum(self.capArray) #caculate the weights by given capacitance array
        v_dac = np.dot(np.array(list(N),dtype=np.float)*self.v_ref, weights) #caculate the dac output
        return v_dac
    
    def synthesizeCapArray(self):
        """function to generate a capacitance array according to given mismatchStd and random seed"""
        capExp = [i for i in range(0, self.n)]
        capExp.reverse()
        capExp.append(0)
        capExp = np.array(capExp)
        nomCap = self.radix**capExp
        np.random.seed(self.seed) #fix the random seed
        if self.flag == 'bad':
            capArray = np.random.normal(nomCap, self.mismatchStd*nomCap)
        elif self.flag == 'good':
            capArray = np.random.normal(nomCap, self.mismatchStd*np.sqrt(nomCap))
        else:
            print("Warning: flag must be set.")
        return capArray

    def complier_and_SAR_logic(self, v_in, v_dac):
        """comlier and SAR logic parts"""
        b = self.sar_logic.get(np.sign(v_dac- v_in), 1)
        return str(b)
    
    def register(self, b, clock, N):
        """register update model, b is the value of current bit that need to update.
            clock is the the current clock value, N is the current digital code.
            The output is the new digital value after update."""
        N_new_l = list(N) #transform the code string into list in order to update the bit value
        N_new_l[clock] = b #change the bit value
        if clock < (len(N)-1): #if it's not the last clock in a period, change next bit into 1.
            N_new_l[clock+1] = '1'
        N_new = ''.join(N_new_l) #transform into string
        return N_new

    def main(self, v_in):
        """SAR_ADC main function, caculate a list of digital output in a period"""
        N='1'+(self.n-1)*'0' #Set the first digital output code
        outputs = [N]
        for i in range(self.n):
            v_dac = self.DAC(N) #caculate the current analog volt
            b = self.complier_and_SAR_logic(v_in, v_dac) #determine the value of current bit(0 or 1)
            N = self.register(b, i, N) #update the digital code
            outputs += [N] #add the current digital code into outputs list
        del outputs[-2]
        return outputs

import numpy as np
import nltk
import matplotlib
import matplotlib.pyplot as plt

"""Code Destiny method part"""
N = 100000
fs = 45e6
f_in = 5.6018e6

v_fs = 1.2
n = 12
v_cm = 0.6
sigma = 0.01
radix = 2
flag = 'good'
seed = 42

sar_adc = SAR_ADC(v_fs, n, v_cm, sigma, radix, flag, seed)

def getDNLs_INLs(N, fs, f_in, sar_dac):
    x = np.linspace(0, 1, fs+1)[:N]
    y_in = (v_fs/2)*np.sin(2*np.pi*f_in*x)+v_cm
    
    y_code_output = [sar_adc.main(y_in_i)[-1] for y_in_i in y_in]
    freqDist = nltk.FreqDist(y_code_output)
    
    dic = {}
    for key in freqDist:
        nCode = np.dot(np.array(list(key),dtype=np.float)*(2**len(key)), np.array([(1/2)**(i+1) for i in range(len(key))]))
        dic[nCode] = freqDist[key]

    code_bins = np.array([dic.get(i,0) for i in range(0, 2**n)])

    N_record_n = np.sum(code_bins[:2**(n-1)])
    N_record_p = np.sum(code_bins[2**(n-1):])

    v_offset = 0.5*sar_adc.amp*np.pi*np.sin((N_record_p-N_record_n)/(N_record_p+N_record_n))
    v_js = [-sar_adc.amp*np.cos(np.pi* ( np.sum(code_bins[:j])/ N) ) for j in range(0, 2**(n))]

    DNLs = [0.0]+ [(v_js[j+1] - v_js[j])*((2**(n))/v_fs)-1 for j in range(0, 2**(n)-1)]
    INLs = [np.sum(DNLs[:j+1]) for j in range(0, 2**(n))]
    return DNLs, INLs

def plot_DNL_INL(dnls, inls):
    x = np.linspace(0, len(dnls), len(dnls)+1)
    
    plt.figure()
    sub_dnls = plt.subplot(1,2,1)
    plt.title('The DNL of given ADC')
    sub_dnls.plot(x, [None]+dnls, linewidth = 0.5)
    plt.ylabel('DNL')
    plt.xlabel('Sequence Number of DNL')
    plt.xlim((0, len(dnls)+1))
    
    plt.grid()
    
    sub_inls = plt.subplot(1,2,2)
    plt.title('The INL of given ADC')
    sub_inls.plot(x, [None]+inls,linewidth = 0.5)
    plt.ylabel('INL')
    plt.xlabel('Sequence Number of INL')
    plt.subplots_adjust(wspace = 0.45)
    plt.grid()
    plt.show()


DNLs, INLs = getDNLs_INLs(N, fs, f_in, sar_adc)
plot_DNL_INL(DNLs, INLs)
