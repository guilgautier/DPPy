from dppy.exotic_dpps import DescentProcess


dp = DescentProcess()

size = 100
dp.sample(size)

dp.plot_vs_bernoullis()
