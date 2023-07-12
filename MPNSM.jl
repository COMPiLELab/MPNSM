# This julia script reproduces the numerical results presented in [Section 6, 1]
#
# [1] K. Bergermann, M. Stoll, and F. Tudisco, A nonlinear spectral core-periphery detection method for multiplex networks, Preprint (2023)
#
# Authors: Kai Bergermann, Francesco Tudisco

push!(LOAD_PATH, pwd());
using LinearAlgebra, SparseArrays, MAT, Plots, Random, ProgressBars, MPNSM_module


#####################################
### Define problem and parameters ###
#####################################

### Available problems:
# Artificial two-layer networks: internet, cardiff, yeast, email
# Real-world multiplex networks: arabidopsis, arxiv, drosophila, european_airlines, homo, rana_plaza_2013, rana_plaza_2014

problem = "cardiff"
noiselevel = 0.25 # only relevant for artificial two-layer networks

alpha = 10
p = 22
q = 2
tol = 1e-08
maxIter = 500

compute_QUBO_objective = true
create_spy_plot = true


# Set irrelevant noise level to zero for real-world multiplex networks
if problem=="arabidopsis" || problem=="arxiv" || problem=="drosophila" || problem=="european_airlines" || problem=="homo" || problem=="rana_plaza_2013" || problem=="rana_plaza_2014"
	noiselevel = 0
end


####################################
### Set up problem and run MPNSM ###
####################################

A, A1, n, L = setup_adjacency_tensor(problem, noiselevel)

println("-------------------------------------------------")
println("Problem:\t", problem)
println("# of nodes:\t", n)
println("# of layers:\t", L)
println("Noise level:\t", noiselevel)
println("-------------------------------------------------")

# Define starting vectors
x0 = ones(Float64, n)
c0 = ones(Float64, L)

# Run [Algorithm 4.1, 1]
x, c = MPNSM(A,x0,c0,alpha,p,q,tol,maxIter)

# Obtain row and column permutation from node coreness vector
ind = sortperm(x,rev=true)
println("Optimized layer coreness vector c:\n", c)
println("-------------------------------------------------")


##############################################################
### Evaluate QUBO objective and plot reordered adjacencies ###
##############################################################

if compute_QUBO_objective
	opt_core_size = compute_QUBO(A, A1, x, c, n, L)
	println("-------------------------------------------------")
end


if create_spy_plot
	if compute_QUBO_objective
		plot_reordered_adjacencies(problem, noiselevel, A1, L, ind, compute_QUBO_objective, p, q, opt_core_size)
	else
		plot_reordered_adjacencies(problem, noiselevel, A1, L, ind, compute_QUBO_objective, p, q)
	end
end


