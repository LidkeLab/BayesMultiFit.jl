var documenterSearchIndex = {"docs":
[{"location":"#BAMF.jl-Documentation","page":"BAMF.jl Documentation","title":"BAMF.jl Documentation","text":"","category":"section"},{"location":"","page":"BAMF.jl Documentation","title":"BAMF.jl Documentation","text":"This repository is currently serving as a vehicle to explore the Julia Language environment using a somewhat sophisticated code base. It implements the core Reversible Jump Markov Chain Monte Carlo (RJMCMC) algorithm from the paper:","category":"page"},{"location":"","page":"BAMF.jl Documentation","title":"BAMF.jl Documentation","text":"Bayesian Multiple Emitter Fitting using Reversible Jump Markov Chain Monte Carlo.   Mohamadreza Fazel, Michael J. Wester, Hanieh Mazloom-Farsibaf, Marjolein B. M. Meddens, Alexandra S. Eklund, Thomas Schlichthaerle, Florian Schueder, Ralf Jungmann & Keith A. Lidke. Scientific Reports, 2019. 9(1): p. 13791.  ","category":"page"},{"location":"","page":"BAMF.jl Documentation","title":"BAMF.jl Documentation","text":"Documentation is rudimentary and should be considered only as proof of principle of the autodoc system at this stage.  ","category":"page"},{"location":"","page":"BAMF.jl Documentation","title":"BAMF.jl Documentation","text":"Modules = [BAMF]","category":"page"},{"location":"#BAMF.ArrayDD","page":"BAMF.jl Documentation","title":"BAMF.ArrayDD","text":"ArrayDD <: BAMFData\n\ndata type for direct detection (i.e. a single image at a camera).   \n\n\n\n\n\n","category":"type"},{"location":"#BAMF.BAMFData","page":"BAMF.jl Documentation","title":"BAMF.BAMFData","text":"BAMFData\n\nBAMFData is an abstract type.  Specific data types will inherit from BAMFData.   The most common example is 'directdetection'  \n\n\n\n\n\n","category":"type"},{"location":"#BAMF.BAMFState","page":"BAMF.jl Documentation","title":"BAMF.BAMFState","text":"BAMFState\n\nBAMFState is an abstract type.  Specific state types will inherit from BAMFState.   States hold the current state of the RJMCMC Chain (i.e. θ) and can also be used to define a true underlying state that generated the data. \n\n\n\n\n\n","category":"type"},{"location":"#BAMF.PSF","page":"BAMF.jl Documentation","title":"BAMF.PSF","text":"PSF\n\nPSF is an abstract type.  Specific psf types will inherit from PSF\n\n\n\n\n\n","category":"type"},{"location":"#BAMF.PSF_airy2D","page":"BAMF.jl Documentation","title":"BAMF.PSF_airy2D","text":"PSF_airy2D <: PSF\n\nAiry PSF is ν²/(4π)(2J₁(νr)/(νr))²   where   ν=πD/(λf)=2π*nₐ/λ  \n\nNote that the Gaussian approximation is σ = 0.42*π/ν\n\n\n\n\n\n","category":"type"},{"location":"#BAMF.RJStruct","page":"BAMF.jl Documentation","title":"BAMF.RJStruct","text":"RJStruct\n\nRJStruct holds the data to be analyzed, the priors, the PSF model,  and the parameters used in the RJMCMC steps. \n\n\n\n\n\n","category":"type"},{"location":"#BAMF.StateFlatBg","page":"BAMF.jl Documentation","title":"BAMF.StateFlatBg","text":"StateFlatBg<: BAMFState\n\nStateFlatBg is an abstract type with childern that implement CPU and GPU variants \n\n\n\n\n\n","category":"type"},{"location":"#BAMF.StateFlatBg_CPU","page":"BAMF.jl Documentation","title":"BAMF.StateFlatBg_CPU","text":"StateFlatBg_CPU(n::Int32, x::Vector{Float32}, y::Vector{Float32}, photons::Vector{Float32}, bg::Float32)\n\nState of a model that has x,y positions, total integrated intensity and a flat background offset. \n\n\n\n\n\n","category":"type"},{"location":"#BAMF.calcintialstate-Tuple{BAMF.RJStruct}","page":"BAMF.jl Documentation","title":"BAMF.calcintialstate","text":"calcintialstate(rjs::RJStruct)\n\ncalculates an intial state by finding the maximum of the data and places an emitter at that location with an intensity selected from the prior.  \n\n\n\n\n\n","category":"method"},{"location":"#BAMF.clusterstates-Tuple{Vector{Any}, Int32}","page":"BAMF.jl Documentation","title":"BAMF.clusterstates","text":"clusterstates(states::Vector{Any},n::Int32)\n\nuses kmeans clustering group x,y locations into n clusters and returns a State_Results structure. \n\n\n\n\n\n","category":"method"},{"location":"#BAMF.genBAMFData-Tuple{BAMF.RJStruct}","page":"BAMF.jl Documentation","title":"BAMF.genBAMFData","text":"genBAMFData(rjs::RJStruct)\n\ngenerate an empty BAMFData stucture for data type in RJStruct\n\n\n\n\n\n","category":"method"},{"location":"#BAMF.getmapnstates-Tuple{Vector{Any}}","page":"BAMF.jl Documentation","title":"BAMF.getmapnstates","text":"states_mapn=getmapn(states::Vector{Any})\n\nreturns a vector of states that where states.n == map_n\n\n\n\n\n\n","category":"method"},{"location":"#BAMF.getxy-Tuple{Vector{Any}}","page":"BAMF.jl Documentation","title":"BAMF.getxy","text":"x,y,photons=getxy(states::Vector{Any})\n\nreturns a vector of all x,y,photons over all states\n\n\n\n\n\n","category":"method"},{"location":"#BAMF.likelihoodratio-Tuple{Array{Float32, N} where N, Array{Float32, N} where N, Array{Float32, N} where N}","page":"BAMF.jl Documentation","title":"BAMF.likelihoodratio","text":"likelihoodratio(m::Array{Float32}, mtest::Array{Float32}, d::Array{Float32})\n\nGeneric Likelihood calculation using Poisson noise statistics\n\n\n\n\n\n","category":"method"},{"location":"#BAMF.poissrnd!-Tuple{Array{Float32, N} where N}","page":"BAMF.jl Documentation","title":"BAMF.poissrnd!","text":"\"     poissrnd!(d::Array{Float32}) In-place Poisson noise corruptor\n\n\n\n\n\n","category":"method"},{"location":"#BAMF.priorpdf-Tuple{BAMF.RJPrior, Any}","page":"BAMF.jl Documentation","title":"BAMF.priorpdf","text":"\"     priorpdf(rjp::RJPrior,θ)   \n\nCalculate PDF(θ)\n\n\n\n\n\n","category":"method"},{"location":"#BAMF.randID-Tuple{Int32}","page":"BAMF.jl Documentation","title":"BAMF.randID","text":"randID(k::Int32)\n\nreturn random number between 1 and k\n\n\n\n\n\n","category":"method"}]
}
