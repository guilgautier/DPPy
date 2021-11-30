from dppy.poissonized_plancherel import PoissonizedPlancherel

theta = 500  # Poisson parameter
pp = PoissonizedPlancherel(theta=theta)
sample = pp.sample()
pp.plot_diagram(sample, normalize=True)
