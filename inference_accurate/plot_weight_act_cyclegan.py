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

a = np.loadtxt("act_G_cycle_bins.txt")

b = np.loadtxt("act_G_cycle_freq.txt")

#a = a[:,:-1] # temporary remove last point of bins list. match dimension for plotting.
print("weight shapes")
print (a.shape)
print (b.shape)

#bins_,freq_ = merg_hist(a, b )
#plt.plot(bins_, freq_, marker="+", color="green")

colors = cm.YlGn(np.linspace(0, 1, 25))

for i in range(24):
    freq_=b[i]
    temp_bins = a[i]
    bins_=(temp_bins[:-1] + temp_bins[1:])/2
    if (i<=2 or i >= 21):
        plt.plot(bins_, freq_,marker="+", color=colors[i], label="conv_"+str(i))
    else :
        plt.plot(bins_, freq_,marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("g_act_new.pdf")
plt.clf()


a = np.loadtxt("act_W_cycle_bins.txt")

b = np.loadtxt("act_W_cycle_freq.txt")


print("weight shapes")
print (a.shape)
print (b.shape)


for i in range(23):
    freq_=b[i]
    temp_bins = a[i]
    bins_=(temp_bins[:-1] + temp_bins[1:])/2
    if (i<=2 or i >= 21):
        plt.plot(bins_, freq_,marker="+", color=colors[i], label="conv_"+str(i))
    else :
        plt.plot(bins_, freq_,marker="+", color=colors[i])
plt.legend(loc="upper left", ncol=1)
plt.savefig ("g_weight_new.pdf")
plt.clf()
