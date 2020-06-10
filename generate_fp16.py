import itertools
import numpy as np
import matplotlib.pyplot as plt
import struct
import math
a = ["".join(seq) for seq in itertools.product("01", repeat=16)]
half_ = []
for item in a:
    half_.append(struct.unpack('!e',struct.pack('!H', int(item, 2)))[0])

half_ = np.array(half_)
print (half_[0])
print (half_[10000])
half_ = half_[~np.isnan(half_)]
half_ = half_[~np.isinf(half_)]
(n_s,bins) =  np.histogram(np.log2(np.absolute(half_[half_ != 0])), bins=20)
plt.plot(bins[:-1],n_s, marker="+", color = "green")
plt.xlabel("log2(representable values)")
plt.ylabel("frequency")
plt.savefig ("fp16.pdf")
plt.show()
#com
