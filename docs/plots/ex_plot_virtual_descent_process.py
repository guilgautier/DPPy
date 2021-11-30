from dppy.descent_processes import VirtualDescentProcess

vdp = VirtualDescentProcess(x0=0.5)

size = 100
sample = vdp.sample(size)

vdp.plot(sample)
