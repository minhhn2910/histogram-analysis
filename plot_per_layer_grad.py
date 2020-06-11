import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.cm as cm

def merg_hist(bins_list, freq_list, n_bins = 10 ): # merg a list of histograms and combine into 1 histogram
    min_range = np.min(bins_list)
    max_range = np.max(bins_list)
    # divide the range into n_bins ;
    step = (max_range - min_range) / n_bins
    bins = np.array([x*step for x in range(0,11)])
    bins = bins + min_range
    freq = [0.0]*10
    #print (bins)
    for idx in range(len(freq_list)):
        bins_ = bins_list[idx]
        freq_ = freq_list[idx]
        for i in range(len(bins_)):
            if ( bins_[i] <= bins [i+1] and  bins_[i] >= bins [i]):
                freq[i] += freq_[i]
    repr_bins = [] # 10 bins for plotting, take average of each period
    for i in range(len(bins) - 1):
        repr_bins.append((bins[i] + bins[i+1])/2.0)

    return repr_bins, freq



a = np.loadtxt("gradient_bins.txt")

b = np.loadtxt("gradient_freq.txt")

a = a[:,:-1] # temporary remove last point of bins list. match dimension for plotting.

print (a.shape)
print (b.shape)

#488750/ 782 /25 = 25
# D1 8 D2 8 G 9
d1_bins = []
d1_freq = []
d2_bins = []
d2_freq = []
g_bins = []
g_freq = []

grads_bins_epoch = np.array(np.vsplit(a, 25))
grads_freq_epoch = np.array(np.vsplit(b, 25))
print (grads_bins_epoch.shape)
print (grads_freq_epoch.shape)

for i in range (25):
    grads_bins_temp = grads_bins_epoch [i]
    grads_freq_temp = grads_freq_epoch [i]
    #inside : 782 iterations
    grads_bins_iteration = np.array(np.vsplit(grads_bins_temp, 782))
    grads_freq_iteration = np.array(np.vsplit(grads_freq_temp, 782))

    for j in range( 782):
        grads_1iter_bins_temp = grads_bins_iteration[j]
        grads_1iter_freq_temp = grads_freq_iteration[j]
        if(len ( grads_1iter_bins_temp) != 25):
            print("err wrong split ")
            exit()
        d1_bins_temp = grads_1iter_bins_temp[:8]
        d1_freq_temp = grads_1iter_freq_temp[:8]
        d2_bins_temp = grads_1iter_bins_temp[8:16]
        d2_freq_temp = grads_1iter_freq_temp[8:16]
        g_bins_temp = grads_1iter_bins_temp[16:]
        g_freq_temp = grads_1iter_freq_temp[16:]
        d1_bins.append(d1_bins_temp)
        d1_freq.append(d1_freq_temp)

        d2_bins.append(d2_bins_temp)
        d2_freq.append(d2_freq_temp)

        g_bins.append(g_bins_temp)
        g_freq.append(g_freq_temp)

print ("shapes ")
d1_bins = np.array(d1_bins)
d2_bins = np.array(d2_bins)
g_bins = np.array(g_bins)
d1_freq = np.array(d1_freq)
d2_freq = np.array(d2_freq)
g_freq = np.array(g_freq)

d1_bins = np.swapaxes(d1_bins,0,1)
d2_bins = np.swapaxes(d2_bins,0,1)
g_bins = np.swapaxes(g_bins,0,1)
d1_freq = np.swapaxes(d1_freq,0,1)
d2_freq = np.swapaxes(d2_freq,0,1)
g_freq = np.swapaxes(g_freq,0,1)

colors = cm.YlGn(np.linspace(0, 1, 25))

for k in range(8):
    temp_bins  = np.array(np.vsplit(d1_bins[k], 25))
    temp_freq  = np.array(np.vsplit(d1_freq[k], 25))


    for i in range(25):
        bins_,freq_ = merg_hist(temp_bins[i], temp_freq[i] )
        if (i<=2 or i >= 22):
            plt.plot(bins_, freq_,marker="+", color=colors[i], label="epoch_"+str(i))
        else :
            plt.plot(bins_, freq_,marker="+", color=colors[i])
    plt.legend(loc="upper left", ncol=1)
    plt.savefig ("d_1_"+ str(k)+".pdf")
    plt.clf()

    temp_bins  = np.array(np.vsplit(d2_bins[k], 25))
    temp_freq  = np.array(np.vsplit(d2_freq[k], 25))


    for i in range(25):
        bins_,freq_ = merg_hist(temp_bins[i], temp_freq[i] )
        if (i<=2 or i >= 22):
            plt.plot(bins_, freq_,marker="+", color=colors[i], label="epoch_"+str(i))
        else :
            plt.plot(bins_, freq_,marker="+", color=colors[i])
    plt.legend(loc="upper left", ncol=1)
    plt.savefig ("d_2_"+ str(k)+".pdf")
    plt.clf()

for k in range(9):

    temp_bins  = np.array(np.vsplit(g_bins[k], 25))
    temp_freq  = np.array(np.vsplit(g_freq[k], 25))


    for i in range(25):
        bins_,freq_ = merg_hist(temp_bins[i], temp_freq[i] )
        if (i<=2 or i >= 22):
            plt.plot(bins_, freq_,marker="+", color=colors[i], label="epoch_"+str(i))
        else :
            plt.plot(bins_, freq_,marker="+", color=colors[i])
    plt.legend(loc="upper left", ncol=1)
    plt.savefig ("g_"+ str(k)+".pdf")
    plt.clf()

print ("shapes ")
print (d1_bins.shape)
print (g_bins.shape)
exit()
