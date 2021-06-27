# heston_nandi_garch

This repo contains the scripts related to the GARCH introduced in "*A Closed Form GARCH Option Valuation Model*" by Heston and Nandi (2000).

Main functions include:
- MLE of HN-GARCH parameters 
- Simulation of future asset prices
- Montecarlo Simulation for option pricing
- Computation of **CDF** and **PDF** through Fourier Transform of the characteristic function

Future development include:
- Computation of Option Price with numerical integration
