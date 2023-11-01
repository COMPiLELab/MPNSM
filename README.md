# MPNSM
A nonlinear spectral core-periphery detection method for multiplex networks

This repository contains julia codes reproducing the numerical results presented in [Section 6, 1].

**Paper:**
[1] K. Bergermann, M. Stoll, and F. Tudisco, A nonlinear spectral core-periphery detection method for multiplex networks, [arXiv:2310.19697](https://arxiv.org/pdf/2310.19697.pdf) (2023)

**Version:**
All codes were tested with julia v1.4.1 on Ubuntu 20.04.6 LTS.

**Packages:**
Standard libraries in julia v1.4.1:
 - LinearAlgebra
 - SparseArrays
 - Random

Packages:
 - MAT v0.10.3
 - Plots v.1.4.3
 - ProgressBars v.1.5.0

This repository contains:

**License:**
 - LICENSE: GNU General Public License v2.0

**Directories:**
 - data: contains .mat files of the adjacency matrices of all networks considered in [1]
 - plots: contains png files of spy plots of reordered adjacency matrices
 
**Main script:**
 - MPNSM.jl: applies [Algorithm 4.1, 1] to specified multiplex network with specified parameters
 
**Module:**
 - MPNSM_module.jl: contains data structure, the implementation of [Algorithm 4.1, 1], other functions called from the main script MPNSM.jl, and helper functions.

**Authors:**
Kai Bergermann (kai.bergermann@math.tu-chemnitz.de), Francesco Tudisco (francesco.tudisco@gssi.it)
