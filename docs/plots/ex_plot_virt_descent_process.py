from exotic_dpps import *



cp = VirtualDescentProcess(size=100,x_0=0.5)

cp.sample()

cp.plot_vs_bernoullis()
