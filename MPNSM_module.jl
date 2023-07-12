
module MPNSM_module

export MPNSM, setup_adjacency_tensor, compute_QUBO, plot_reordered_adjacencies

using LinearAlgebra, SparseArrays, MAT, Plots, Random, ProgressBars

######################
### Data structure ###
######################

struct Sparse3Tensor
	# Customized data type for (n by n by L) adjacency tensors of multiplex networks in (I,J,K,V)-format
	# with	fields	I:	row index,
	#		J:	column index,
	#		K:	layer index,
	#		V:	value of tensor entry,
	#		n:	number of nodes per layer,
	#		L:	number of layers.
	
	I::Vector{Int64}
	J::Vector{Int64}
	K::Vector{Int64}
	V::Vector{Float64}
	n::Int64
	L::Int64
end


###########################################
### [Algorithm 4.1, 1] -- main function ###
###########################################


function MPNSM(A::Sparse3Tensor, x0::Vector{Float64}, c0::Vector{Float64}, a, p, q, tol, maxIter)
	# Computes an optimal node and layer coreness vector for a given multiplex network.
	#
	# Input:	A: 		third-order adjacency tensor of the multiplex network in customized data format,
	#		x0: 		positive initial node coreness vector,
	#		c0:		positive initial layer coreness vector,
	#		a:		alpha-parameter (>1) in the smoothed maximum (kernel) function,
	#		p:		norm-parameter (>1) on the node coreness vector x,
	#		q:		norm-parameter (>1) on the layer coreness vector x,
	#		tol:		tolerance between consecutive iterates,
	#		maxIter:	maximum number of iterations in case tol is not reached before.
	#
	# Output:	x:		optimized node coreness vector, normalized in p-norm,
	#		c:		optimized layer coreness vector, normalized in q-norm.
	
	pp = p/(p-1)
	qq = q/(q-1)

	x0 = x0/norm(x0,pp)
	c0 = c0/norm(c0,qq)
	
	x = x0
	c = c0
	
        println("Nonlinear Spectral Method for Multiplex Networks:")
        println("-------------------------------------------------")
        println("alpha:\t\t$a\np:\t\t$p\nq:\t\t$q\ntol:\t\t$tol")
	
	t = @elapsed begin
		for i = 1:maxIter
			x = gradfx(A,x0,c0,a)
			x = x/norm(x,pp)
			x = x.^(1/(p-1))
					
			c = gradfc(A,x0,a)
			c = c/norm(c,qq)
			c = c.^(1/(q-1))
			
			#@printf("Iteration %d, x error norm: %e, c error norm: %e\n", i, norm(x-x0), norm(c-c0))
			
			if norm(x-x0)<tol && norm(c-c0)<tol
				println("Num iter:\t$i")
				break
			else
				x0 = x
				c0 = c
			end
		end
	end
	println("Exec time:\t$t sec")
	println("-------------------------------------------------")
		
	return x, c
end


###################################################
### Other functions called from the main script ###
###################################################


function setup_adjacency_tensor(problem::String, noiselevel=0.1)
	# Builds the multiplex network of the problems considered in [Section 6, 1].
	#
	# Available problems:
	# Artificial two-layer networks: internet, cardiff, yeast, email
	# Real-world multiplex networks: arabidopsis, arxiv, drosophila, european_airlines,
	#				homo, rana_plaza_2013, rana_plaza_2014
	#
	# Input: 	problem:	one of the above strings,
	#		noiselevel:	for artificial two-layer networks only! Portion of the
	#				number of edges in the informative layer to be placed
	#				in the (second) random noise layer.
	#
	# Output:	A:		Sparse3Tensor version of the third-order adjacency tensor,
	#		A1:		array of layer adjacency matrices in CSC format,
	#		n:		number of nodes per layer,
	#		L:		number of layers.

	
	Random.seed!(1234)
	
	if problem=="internet"
		vars = matread("data/internet2006-autonomous_systems(as-22july06).mat")
		A_inf = vars["Problem"]["A"]
		n = size(A_inf,1)
		L = 2

		B = sprand(n,n,noiselevel*nnz(A_inf)/n^2).>0
		A_noise = triu(B)+triu(B,1)'
		
		A1 = SparseMatrixCSC{Float64,Int64}[]
		push!(A1, A_inf)
		push!(A1, A_noise)

		I1, J1, V1 = findnz(A_inf)
		K1 = ones(length(I1))
		I2, J2, V2 = findnz(A_noise)
		K2 = 2*ones(length(I2))

		A = Sparse3Tensor(vcat(I1, I2), vcat(J1, J2), vcat(K1, K2), vcat(V1, V2), n, L)
	
	elseif problem=="cardiff"
		vars = matread("data/Cardiff.mat")
		A_inf = vars["Problem"]["A"]
		n = size(A_inf,1)
		L = 2

		B = sprand(n,n,noiselevel*nnz(A_inf)/n^2).>0
		A_noise = triu(B)+triu(B,1)'
		
		A1 = SparseMatrixCSC{Float64,Int64}[]
		push!(A1, A_inf)
		push!(A1, A_noise)

		I1, J1, V1 = findnz(A_inf)
		K1 = ones(length(I1))
		I2, J2, V2 = findnz(A_noise)
		K2 = 2*ones(length(I2))

		A = Sparse3Tensor(vcat(I1, I2), vcat(J1, J2), vcat(K1, K2), vcat(V1, V2), n, L)
		
	elseif problem=="yeast"
		vars = matread("data/yeast.mat")
		A_inf = vars["Problem"]["A"]
		n = size(A_inf,1)
		L = 2

		B = sprand(n,n,noiselevel*nnz(A_inf)/n^2).>0
		A_noise = triu(B)+triu(B,1)'
		
		A1 = SparseMatrixCSC{Float64,Int64}[]
		push!(A1, A_inf)
		push!(A1, A_noise)

		I1, J1, V1 = findnz(A_inf)
		K1 = ones(length(I1))
		I2, J2, V2 = findnz(A_noise)
		K2 = 2*ones(length(I2))

		A = Sparse3Tensor(vcat(I1, I2), vcat(J1, J2), vcat(K1, K2), vcat(V1, V2), n, L)
		
	elseif problem=="email"
		vars = matread("data/email-EuAll_max_conn_comp.mat")
		A_inf = vars["A"]
		n = size(A_inf,1)
		L = 2

		B = sprand(n,n,noiselevel*nnz(A_inf)/n^2).>0
		A_noise = triu(B)+triu(B,1)'
		
		A1 = SparseMatrixCSC{Float64,Int64}[]
		push!(A1, A_inf)
		push!(A1, A_noise)

		I1, J1, V1 = findnz(A_inf)
		K1 = ones(length(I1))
		I2, J2, V2 = findnz(A_noise)
		K2 = 2*ones(length(I2))

		A = Sparse3Tensor(vcat(I1, I2), vcat(J1, J2), vcat(K1, K2), vcat(V1, V2), n, L)
		
	elseif problem=="arabidopsis"
		vars = matread("data/arabidopsis_adjacencies.mat")
		n = size(vars["A_single"][1],1)
		L = 7

		A1 = vars["A_single"]
		for l=1:L
			A1[l] = A1[l].>0
		end
        
		A = build_multiplex_tensor(A1, n, L)
	
	elseif problem=="arxiv"
		vars = matread("data/arxiv_adjacencies.mat")
		n = size(vars["A_single"][1],1)
		L = 13

		A1 = vars["A_single"]
		for l=1:L
			A1[l] = A1[l].>0
		end
		
		A = build_multiplex_tensor(A1, n, L)
		
	elseif problem=="drosophila"
		vars = matread("data/drosophila_adjacencies.mat")
		n = size(vars["A_single"][1],1)
		L = 7

		A1 = vars["A_single"]
		for l=1:L
			A1[l] = A1[l].>0
		end
        
		A = build_multiplex_tensor(A1, n, L)
		
	elseif problem=="european_airlines"
		vars = matread("data/multiplex_airlines_GC.mat")
		n = size(vars["net"]["A"][1],1)
		L = 37

		A1 = vars["net"]["A"]
		for l=1:L
			A1[l] = A1[l].>0
		end
        
		A = build_multiplex_tensor(A1, n, L)
    
	elseif problem=="homo"
		vars = matread("data/homo_adjacencies.mat")
		n = size(vars["A_single"][1],1)
		L = 7

		A1 = vars["A_single"]
		for l=1:L
			A1[l] = A1[l].>0
		end
        
		A = build_multiplex_tensor(A1, n, L)
		
	elseif problem=="rana_plaza_2013"
		Al1 = matread("data/rana_plaza_20130417-20130511_Aintra_retweet.mat")
		Al2 = matread("data/rana_plaza_20130417-20130511_Aintra_reply.mat")
		Al3 = matread("data/rana_plaza_20130417-20130511_Aintra_mention.mat")
		
		n = size(Al1["Aintra_retweet"],1)
		L = 3
		
		Al1 = (0.5*(Al1["Aintra_retweet"] + Al1["Aintra_retweet"]')).>0
		Al2 = (0.5*(Al2["Aintra_reply"] + Al2["Aintra_reply"]')).>0
		Al3 = (0.5*(Al3["Aintra_mention"] + Al3["Aintra_mention"]')).>0
		
		A1 = SparseMatrixCSC{Float64,Int64}[]
		push!(A1, Al1)
		push!(A1, Al2)
		push!(A1, Al3)
		
		I1, J1, V1 = findnz(Al1)
		K1 = ones(length(I1))
		I2, J2, V2 = findnz(Al2)
		K2 = 2*ones(length(I2))
		I3, J3, V3 = findnz(Al3)
		K3 = 3*ones(length(I3))

		A = Sparse3Tensor(vcat(I1, I2, I3), vcat(J1, J2, J3), vcat(K1, K2, K3), vcat(V1, V2, V3), n, L)
		
	elseif problem=="rana_plaza_2014"
		Al1 = matread("data/rana_plaza_20140417-20140511_Aintra_retweet.mat")
		Al2 = matread("data/rana_plaza_20140417-20140511_Aintra_reply.mat")
		Al3 = matread("data/rana_plaza_20140417-20140511_Aintra_mention.mat")
		
		n = size(Al1["Aintra_retweet"],1)
		L = 3
		
		Al1 = (0.5*(Al1["Aintra_retweet"] + Al1["Aintra_retweet"]')).>0
		Al2 = (0.5*(Al2["Aintra_reply"] + Al2["Aintra_reply"]')).>0
		Al3 = (0.5*(Al3["Aintra_mention"] + Al3["Aintra_mention"]')).>0
		
		A1 = SparseMatrixCSC{Float64,Int64}[]
		push!(A1, Al1)
		push!(A1, Al2)
		push!(A1, Al3)
		
		I1, J1, V1 = findnz(Al1)
		K1 = ones(length(I1))
		I2, J2, V2 = findnz(Al2)
		K2 = 2*ones(length(I2))
		I3, J3, V3 = findnz(Al3)
		K3 = 3*ones(length(I3))

		A = Sparse3Tensor(vcat(I1, I2, I3), vcat(J1, J2, J3), vcat(K1, K2, K3), vcat(V1, V2, V3), n, L)

	end
	
	return A, A1, n, L
end


function compute_QUBO(A, A1, x, c, n, L)
	# Implements the sweeping procedure to compute the QUBO objective function value
	# described in [Section 5, 1]. Prints the optimal core size alongside the corresponding
	# QUBO objective function value. Returns the optimal core size s*.
	#
	# Input:	A:		adjacency tensor in Sparse3Tensor format,
	#		A1:		array of layer adjacency matrices in CSC format,
	#		x:		optimized node coreness vector to be sweeped over,
	#		c:		optimized layer coreness vector,
	#		n:		number of nodes per layer,
	#		L:		number of layers.
	#
	# Output:	opt_core_size:	optimal core size s* obtained by the sweeping procedure.
	
	N1 = zeros(L); N2 = zeros(L)
	for k in 1:L
		N1[k] = edges_of_layer(A,k) # present edges in layer k
		N2[k] = n*n - N1[k] # missing edges in layer k
	end

	fff(x) = eval_qubo_f(A1,x,c,N1,N2)
	opt_core_size, opt_QUBO_value = sweep_binary_vector(x, fff)
	
	println("-------------------------------------------------")
	println("Optimal QUBO index s*: ", opt_core_size, "\nOptimal QUBO objective function value: ", opt_QUBO_value)
	
	return opt_core_size
end


function plot_reordered_adjacencies(problem, noiselevel, A1, L, ind, compute_QUBO_objective, p, q, opt_core_size=0)
	# Visualizes the obtained core-periphery partition by means of reordered adjacency
	# matrix spy plots. For the artificial two-layer networks from [Section 6.1, 1],
	# only the first informative layer is plotted and saved in png format. For the
	# real-world multiplex networks from [Section 6.2, 1], all layers are plotted
	# and saved in png format.
	#
	# Input:	problem:			string of the network name required for filename,
	#		noiselevel:			for artificial two-layer networks only! required
	#						for filename,
	#		A1:				layer adjacency matrices to be spy-plotted with
	#						permuted rows and columns,
	#		L:				number of layers,
	#		ind:				permutation vector for rows and columns according
	#						to the descendingly sorted node coreness vector,
	#		compute_QUBO_objective:	binary. If true, marks optimal core size s* by red lines,
	#		p:				norm-parameter on x required for filename,
	#		q:				norm-parameter on c required for filename,
	#		opt_core_size:			required if compute_QUBO_objective==true. Optimal core
	#						size s* computed by sweeping procedure.
	#
	# Output:	None. But the produced spy plot is stored in png format in "plots/" directory.
	
	gr()
	if problem == "internet" || problem == "cardiff" || problem == "yeast" || problem == "email"
		A = A1[1]
		spy(A[ind, ind],  markersize=1.5, xaxis=nothing, yaxis=nothing, framestyle = :box, legend = nothing, markercolor = :lighttest, markerstrokecolor = :lighttest)
		if compute_QUBO_objective
			plot!([opt_core_size], seriestype="vline", linewidth=3, linecolor=:red)
			plot!([opt_core_size], seriestype="hline", linewidth=3, linecolor=:red)
		end
		savefig(string("plots/",problem,"_noise_",noiselevel,"_p_",p,"_q_",q,".png"))

	else
		plt = Plots.Plot{Plots.GRBackend}[]
		for l=1:L
			pl = spy(A1[l][ind, ind].>0,  markersize=1.5, xaxis=nothing, yaxis=nothing, framestyle = :box, legend = nothing, markercolor = :lighttest, markerstrokecolor = :lighttest)
			if compute_QUBO_objective
				plot!([opt_core_size], seriestype="vline", linewidth=3, linecolor=:red)
				plot!([opt_core_size], seriestype="hline", linewidth=3, linecolor=:red)
			end
		
		push!(plt, pl)
		end
		plot(plt...)
		savefig(string("plots/",problem,"_p_",p,"_q_",q,".png"))
	end
end


########################
### Helper functions ###
########################


function edges_of_layer(A::Sparse3Tensor, layer_number)
	# Computes the number of edges present in specified layer.
	#
	# Input:	A:		adjacency tensor in Sparse3Tensor format,
	#		layer_number:	layer number of the adjacency tensor A.
	#
	# Output:	edge_count:	number of edges in specified layer.
	
	edge_count = 0
	for (i, j, k, v) in zip(A.I, A.J, A.K, A.V)
		if k == layer_number
			edge_count += 1
		end
	end
	return edge_count
end


function sweep_binary_vector(x::Vector{Float64}, fun)
	# Implements the sweeping procedure for computing multiplex QUBO objective function values,
	# see [Section 5, 1], i.e., given a node coreness vector x, [Equation (5.3), 1] is evaluated
	# for the length(x) binary vectors with one-entries in the first k entries and zeros otherwise.
	#
	# Input:	x:		node coreness vector optimized by [Algorithm 4.1, 1],
	#		fun: 		implements the evaluation of [Equation (5.3), 1] for any given
	#				binary vector y.
	#
	# Output:	bestindex:	optimal core size s*,
	#		bestval:	QUBO objective function value corresponding to s* computed by
	#				[Equation (5.3), 1].
	
	y = zeros(length(x))
	p = sortperm(x,rev=true)
	bestval = 0
	bestindex = 1
	println("Progress on QUBO sweep:")
	for k in tqdm(1:length(x))
		y[p[k]] = 1
		fy = fun(y)
		if fy > bestval
			bestval = fy
			bestindex = k
		end
	end
	
	return bestindex, bestval
end


function eval_qubo_f(A1, x::Vector{Float64}, c::Vector{Float64}, N1::Vector{Float64}, N2::Vector{Float64})
	# Routine used within 'sweep_binary_vector' to evaluate [Equation (5.3), 1].
	#
	# Input:	A1:	adjacency tensor in CSC-format (A1[l] contains the adjacency matrix of layer l)
	#		x:	a given binary vector from the sweeping procedure,
	#		c:	optimized layer coreness vector c (fixed for all binary vectors x),
	#		N1:	vector of present edges in the layers,
	#		N2:	vector of missing edges in the layers.
	#
	# Output:	f:	value of [Equation (5.3), 1] for given input.
	
	n = size(A1[1],1)
	L = length(N1)
	f = 0
	
	norm_c = norm(c,1)
	
	for l=1:L
		Dl = spdiagm(0 => sum(A1[l], dims=2)[:,1])
	
		f += c[l]/norm_c * (x'*(2*(1/N1[l] + 1/N2[l])*Dl - 2*(n-1)*(1/N2[l])*spdiagm(0 => ones(n)) - (1/N1[l] + 1/N2[l])*A1[l])*x + (1/N2[l])*(sum(x)^2 - sum(x)))
	end
		
	return f
end


function fFx(x, y, a)
	# Helper function for evaluating grad_x of the objective function f
	# defined in [Equation (2.1), 1].
	#
	# Input:	x:	one vector entry,
	#		y:	another vector entry,
	#		a:	scalar alpha from the smoothed maximum (kernel) function.
	#
	# Output:	f:	value of expression.
	
	f = x^(a-1) * (x^a + y^a)^(1/a-1)
	return f
end


function fFc(x, y, a)
	# Helper function for evaluating the objective function f and grad_c
	# of the objective function f defined in [Equation (2.1), 1].
	#
	# Input:	x:	one vector entry,
	#		y:	another vector entry,
	#		a:	scalar alpha from the smoothed maximum (kernel) function.
	#
	# Output:	f:	value of expression.
	
	f = (x^a + y^a)^(1/a)
	return f
end


function gradfc(A::Sparse3Tensor, x::Vector{Float64}, a)
	# Computes the gradient of the objective function f defined
	# in [Equation (2.1), 1] w.r.t. c.
	#
	# Input:	A:	adjacency tensor in Sparse3Tensor format,
	#		x:	node coreness vector x,
	#		a:	scalar alpha from the smoothed maximum (kernel) function.
	#
	# Output:	Fc:	desired gradient of f w.r.t. c.
	
	Fc = zeros(Float64, A.L)

	for (i, j, k, v) in zip(A.I, A.J, A.K, A.V)
		@inbounds Fc[k] += v * fFc(x[i], x[j], a)
	end
	
	return Fc
end


function gradfx(A::Sparse3Tensor, x::Vector{Float64}, c::Vector{Float64}, a)
	# Computes the gradient of the objective function f defined
	# in [Equation (2.1), 1] w.r.t. x.
	#
	# Input:	A:	adjacency tensor in Sparse3Tensor format,
	#		x:	node coreness vector x,
	#		c:	layer coreness vector c,
	#		a:	scalar alpha from the smoothed maximum (kernel) function.
	#
	# Output:	Fx:	desired gradient of f w.r.t. x.
	
	Fx = zeros(Float64, A.n)

	for (i, j, k, v) in zip(A.I, A.J, A.K, A.V)
		@inbounds Fx[i] += 2 * v * c[k] * fFx(x[i], x[j], a)
	end
	
	return Fx
end


function eval_f(A::Sparse3Tensor, x::Vector{Float64}, c::Vector{Float64}, a)
	# Evaluates the objective function f defined in [Equation (2.1), 1].
	#
	# Input:	A:	adjacency tensor A in Sparse3Tensor format,
	#		x:	node coreness vector,
	#		c:	layer coreness vector,
	#		a:	scalar alpha from the smoothed maximum (kernel) function.
	#
	# Output:	f:	desired objective function value.
	
	f = 0
	
	for (i, j, k, v) in zip(A.I, A.J, A.K, A.V)
		@inbounds f += v * c[k] * fFc(x[i], x[j], a)
	end
	
	return f
end


function build_multiplex_tensor(A1, n, L)
	# Transforms the array of adjacency matrices in CSC format into the
	# customized Sparse3Tensor format.
	#
	# Input:	A1:	array of adjacency matrices in CSC format,
	#		n:	number of nodes per layer,
	#		L:	number of layers.
	#
	# Output:	A:	adjacency tensor in Sparse3Tensor format.

	curI, curJ, curV = findnz(A1[1])
	curK = ones(length(curI))
	
	for l=2:L
		I, J, V = findnz(A1[l])
		K = l*ones(length(I))
		
		curI = vcat(curI, I)
		curJ = vcat(curJ, J)
		curK = vcat(curK, K)
		curV = vcat(curV, V)
	end
	
	A = Sparse3Tensor(curI, curJ, curK, curV, n, L)
	
	return A
end

end

