import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.cm as cm
a = np.loadtxt("w_g_bins.txt")

b = np.loadtxt("w_g_freq.txt")

a = a[:,:-1] # temporary remove last point of bins list. match dimension for plotting.


d_1_1 = a [::3]
d_1_2 = b [::3]
d_2_1 = a [1::3]
d_2_2 = b [1::3]
g_1_1 = a [2::3]
g_1_2 = b [2::3]
#data written to file: Gradient histogram of D_real, histogram of D_fake, histogram of G_
print (d_1_1.shape)
#19550 / 25 (epoch ) = 782 histogram to merge for 1 epoch
print (d_2_1.shape)
print (g_1_1.shape)

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


epochs_d_1_1 = np.array(np.vsplit(d_1_1, 25))
epochs_d_1_2 = np.array(np.vsplit(d_1_2, 25))
epochs_d_2_1 = np.array(np.vsplit(d_2_1, 25))
epochs_d_2_2 = np.array(np.vsplit(d_2_2, 25))
epochs_g_1_1 = np.array(np.vsplit(g_1_1, 25))
epochs_g_1_2 = np.array(np.vsplit(g_1_2, 25))
bins_d_1 = []
freq_d_1 =  []
bins_d_2 = []
freq_d_2 = []
bins_g_1 = []
freq_g_1 = []
for i in range(25):
    bins,freq = merg_hist(epochs_d_1_1[i], epochs_d_1_2[i] )
    bins_d_1.append(bins)
    freq_d_1.append(freq)
    bins,freq = merg_hist(epochs_d_2_1[i], epochs_d_2_2[i] )
    bins_d_2.append(bins)
    freq_d_2.append(freq)
    bins,freq = merg_hist(epochs_g_1_1[i], epochs_g_1_2[i] )
    bins_g_1.append(bins)
    freq_g_1.append(freq)

#print (bins_d_1)
print ("plotting d1 ")
colors = cm.YlGn(np.linspace(0, 1, len(bins_d_1)))
for i in range(25):
    if (i<=2 or i >= 22):
        plt.plot(bins_d_1[i], freq_d_1[i],marker="+", color=colors[i], label="epoch_"+str(i))
    else :
        plt.plot(bins_d_1[i], freq_d_1[i],marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("d_1.pdf")
plt.clf()

print ("plotting d2 ")
colors = cm.YlGn(np.linspace(0, 1, len(bins_d_2)))
for i in range(25):
    if (i<=2 or i >= 22):
        plt.plot(bins_d_2[i], freq_d_2[i],marker="+", color=colors[i], label="epoch_"+str(i))
    else :
        plt.plot(bins_d_2[i], freq_d_2[i],marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("d_2.pdf")
plt.clf()

print ("plotting g1 ")
colors = cm.YlGn(np.linspace(0, 1, len(bins_g_1)))
for i in range(25):
    if (i<=2 or i >= 22):
        plt.plot(bins_g_1[i], freq_g_1[i],marker="+", color=colors[i], label="epoch_"+str(i))
    else :
        plt.plot(bins_g_1[i], freq_g_1[i],marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("g_1.pdf")
plt.clf()
#bins,freq = merg_hist(d_1_1[:2], d_1_2[:2] )
