from exotic_dpps import *

base = 10 # base
cp = CarriesProcess(base)

size = 100
cp.sample(size)

cp.plot_vs_bernoullis()
plt.show()