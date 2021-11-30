from dppy.descent_processes import DescentProcess

dp = DescentProcess()

size = 100
sample = dp.sample(size)

dp.plot(sample)
