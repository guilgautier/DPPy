from exotic_dpps import *

size = 100

cp = DescentProcess(base,size)

cp.sample()

cp.plot_vs_bernoullis()
