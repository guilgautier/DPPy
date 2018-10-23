from exotic_dpps import *

vdp = VirtualDescentProcess(x_0=0.5)

size=100
vdp.sample(size)

vdp.plot_vs_bernoullis()
plt.show()