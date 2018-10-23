from exotic_dpps import *

dp = DescentProcess()

size = 100
dp.sample(size)

dp.plot_vs_bernoullis()
plt.show()
