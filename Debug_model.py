"""Debug model"""

SAR_ADC_model = SAR_ADC(v_fs=1.2, n=12, v_cm=0.6, mismatchStd=0, radix=2, flag='good', seed=31)


def plot(sar_adc, periods):
    """function to plot the """
    x= np.arange(0, periods*n+1, n) #the x axis point of v_in
    x1 = np.arange(0, periods*n+1, 1) #the x axis potin of v_dac, v_x
    v_in = sar_adc.amp*np.sin(2*np.pi*0.01*x) + sar_adc.amp  #define the input signal V_in
    v_x = [v_cm] #first value of V_x
    v_dac = [sar_adc.v_ref/2] #first value of V_dac
    
    #caculate V_dac and V_x value in every periods
    for v_in_i in v_in[1:]:
        v_dac_i = [sar_adc.DAC(N) for N in sar_adc.main(v_in_i)] #caculate the V_dac in every clock in 1 periods
        v_x_i = list(map(lambda i: i-v_in_i+v_cm, v_dac_i))  #caculate the V_x in every clock in 1 periods
        v_dac += v_dac_i #add the i-th v_dac into list v_dac
        v_x += v_x_i #add the i-th v_dac into list v_dac
    
    
    #plot
    plt.figure()
    #plot first sub figure for V_dac and V_in
    sub1 = plt.subplot(2,1,1)
    plt.title('The relation between V_in and V_DAC')
    sub1.step(x1, v_dac,  color='red', label='V_DAC') #plot the curve of V_DAC-V_cm
    sub1.step(x, v_in,  color='blue', label='V_in') #lot the curve of V_in
    plt.ylabel('V/V')
    plt.xlabel('time/s')
    plt.xticks([0]+[i for i in range(n, periods*n+1, n)], [0]+[i for i in range(1,periods+1)])
    #plt.xticks([0]+[i-n/2 for i in range(n, periods*n+1, n)], [0]+[i for i in range(1,periods+1)]) #mapping the original x axis value to period
    for i in range(0, periods*n+1, n): #plot the gridding
        plt.axvline([i],hold=None,linewidth=1,color='grey',linestyle="--")
    plt.legend(loc='lower right', prop={'size': 6})

    #plot the second subfigure for V_x
    sub2 = plt.subplot(2,1,2)
    plt.title('The relation between V_x and V_cm')
    sub2.step(x1, v_x,  color='red', label='V_x')
    plt.axhline([v_cm],hold=None,linewidth=1,color='green',linestyle="--",label='V_cm')
    plt.xticks([0]+[i for i in range(n, periods*n+1, n)], [0]+[i for i in range(1,periods+1)])
    #plt.xticks([0]+[i-n/2 for i in range(n, periods*n+1, n)], [0]+[i for i in range(1,periods+1)]) #to show periods
    for i in range(0, periods*n+1, n):
        plt.axvline([i],hold=None,linewidth=1,color='grey',linestyle="--")
    plt.legend(loc='lower right', prop={'size': 6})
    plt.xlabel('time/s')
    plt.ylabel('V/V')
    plt.subplots_adjust(hspace = 1)
    plt.show()

plot(sar_adc = SAR_ADC_model, periods= 10)

