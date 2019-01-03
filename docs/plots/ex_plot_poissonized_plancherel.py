from dppy.exotic_dpps import PoissonizedPlancherel

theta = 500  # Poisson parameter
pp_dpp = PoissonizedPlancherel(theta=theta)
pp_dpp.sample()
pp_dpp.plot_diagram(normalization=True)
