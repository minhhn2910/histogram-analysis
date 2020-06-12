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


colors = cm.YlGn(np.linspace(0, 1, 25))

a = np.loadtxt("act_d_bins.txt")

b = np.loadtxt("act_d_freq.txt")
a = a[:15640]
b = b[:15640]
a = a[:,:-1] # temporary remove last point of bins list. match dimension for plotting.
print("weight shapes")
print (a.shape)
print (b.shape)

a = np.array(np.vsplit(a, 20))
b = np.array(np.vsplit(b, 20))

for i in range(20):
    bins_,freq_ = merg_hist(a[i], b[i] )
    if (i<=2 or i >= 17):
        plt.plot(bins_, freq_,marker="+", color=colors[i], label="epoch_"+str(i))
    else :
        plt.plot(bins_, freq_,marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("act_g.pdf")
plt.clf()

a = np.loadtxt("act_g_bins.txt")

b = np.loadtxt("act_g_freq.txt")
a = a[:15640]
b = b[:15640]
a = a[:,:-1] # temporary remove last point of bins list. match dimension for plotting.

a = np.array(np.vsplit(a, 20))
b = np.array(np.vsplit(b, 20))

for i in range(20):
    bins_,freq_ = merg_hist(a[i], b[i] )
    if (i<=2 or i >= 17):
        plt.plot(bins_, freq_,marker="+", color=colors[i], label="epoch_"+str(i))
    else :
        plt.plot(bins_, freq_,marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("act_d.pdf") ##wrong naming
plt.clf()

a = np.loadtxt("w_bins.txt")

b = np.loadtxt("w_freq.txt")

a = a[:,:-1] # temporary remove last point of bins list. match dimension for plotting.

print("weight shapes")
print (a.shape)
print (b.shape)
g_weight_bins = a[::2]
g_weight_freq = b[::2]
d_weight_bins = a[1::2]
d_weight_freq = b[1::2]


g_weight_bins = np.array(np.vsplit(g_weight_bins, 20))
g_weight_freq = np.array(np.vsplit(g_weight_freq, 20))
d_weight_bins = np.array(np.vsplit(d_weight_bins, 20))
d_weight_freq = np.array(np.vsplit(d_weight_freq, 20))

for i in range(20):
    bins_,freq_ = merg_hist(g_weight_bins[i], g_weight_freq[i] )
    if (i<=2 or i >= 17):
        plt.plot(bins_, freq_,marker="+", color=colors[i], label="epoch_"+str(i))
    else :
        plt.plot(bins_, freq_,marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("weight_g_.pdf") ##wrong naming
plt.clf()

for i in range(20):
    bins_,freq_ = merg_hist(d_weight_bins[i], d_weight_freq[i] )
    if (i<=2 or i >= 17):
        plt.plot(bins_, freq_,marker="+", color=colors[i], label="epoch_"+str(i))
    else :
        plt.plot(bins_, freq_,marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("weight_d_.pdf") ##wrong naming
plt.clf()
