from dppy.exotic_dpps import VirtualDescentProcess


vdp = VirtualDescentProcess(x_0=0.5)

size = 100
vdp.sample(size)

vdp.plot(vs_bernoullis=True)
