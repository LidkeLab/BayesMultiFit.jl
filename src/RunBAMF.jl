#Demonstrates use of BAMF on a single region

include("RJMCMC.jl")
include("BAMF.jl")

using .RJMCMC

#setup the RJMCMC model
njumptypes=3
jumpprobability=[1,2,3]
jumpprobability=jumpprobability/sum(jumpprobability)

#create a structure with all model info
iterations=10
burnin=20
myrjs=RJMCMC.rjstruct(burnin,iterations,njumptypes,jumpprobability)

mychain=RJMCMC.buildchain(myrjs)


