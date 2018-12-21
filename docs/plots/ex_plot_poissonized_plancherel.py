from dppy.exotic_dpps import PoissonizedPlancherel

theta = 150  # Poisson parameter
pp_dpp = PoissonizedPlancherel(theta=theta)
pp_dpp.sample()
pp_dpp.plot()
