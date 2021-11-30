from dppy.descent_processes import CarriesProcess

base = 10  # base
cp = CarriesProcess(base)

size = 100
sample = cp.sample(size)

cp.plot(sample)
