from dppy.exotic_dpps import PoissonizedPlancherel


theta = 500  # Poisson parameter
pp = PoissonizedPlancherel(theta=theta)
pp.sample()
pp.plot_diagram(normalization=True)
