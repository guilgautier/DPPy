from dppy.exotic_dpps import CarriesProcess


base = 10  # base
cp = CarriesProcess(base)

size = 100
cp.sample(size)

cp.plot(vs_bernoullis=True)
