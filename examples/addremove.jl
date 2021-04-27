## This shows BAMF with only an add/remove option for the RJMCMC chain

using Revise
using Plots
println(pwd())
include("../src/RJMCMC.jl")
include("../src/BAMF.jl")
using .RJMCMC
#using .BAMF
using ImageView
ImageView.closeall()


## make prior on emitter intensity distributions
using Distributions
α=Float32(2)
θ=Float32(500)
g=Gamma(α,θ)
len=Int32(1024)
θ_start=Float32(1)
θ_step=Float32(2.0)
x=range(θ_start,step=θ_step,length=len)
mypdf=pdf(g,x)
plot(x,mypdf)
prior_photons=BAMF.RJPrior(len,θ_start,θ_step,mypdf)


## create a dataset 
n=Int32(3)
sz=Int32(16)
sigma=1.3f0
x=Vector{Float32}([sz/2,sz/2,sz/2])+randn(Float32,3)
y=Vector{Float32}([sz/2,sz/2,sz/2])+randn(Float32,3)
photons=Float32.(rand(g,3))
bg=1f-4
datastate=BAMF.StateFlatBg(n,x,y,photons,bg)

##  create a model then make noisy
dataroi=BAMF.ArrayDD(sz)
BAMF.genmodel_2Dgauss!(datastate,sz,sigma,dataroi.data)
noisyroi=BAMF.poissrnd(dataroi)


## create a BAMF-type RJMCMC structure
xystd=sigma/10;
istd=10;
bndpixels=2
myRJ=BAMF.RJStructDD(sz,sigma,xystd,istd,noisyroi,bndpixels,prior_photons)

## setup the RJMCMC.jl model
#Jumptypes are: move, bg, add, remove, split, merge
njumptypes=6
jumpprobability=[1,0,.01,0,0,0] #Move only
jumpprobability=jumpprobability/sum(jumpprobability)

# create an RJMCMC structure with all model info
iterations=5000
burnin=500
acceptfuns=[BAMF.accept_move,BAMF.accept_bg,BAMF.accept_add] #array of functions
propfuns=[BAMF.propose_move,BAMF.propose_bg,BAMF.propose_add] #array of functions
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)

#create an intial state
state1=BAMF.calcintialstate(myRJ)
n=Int32(3)
sz=Int32(16)
sigma=1.3f0
x=Vector{Float32}([sz/2,sz/2,sz/2])+randn(Float32,3)
y=Vector{Float32}([sz/2,sz/2,sz/2])+randn(Float32,3)
photons=Vector{Float32}([777,500,800])
bg=1f-4


## run chain
@time mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1)

## Display
zoom=Int32(20)
jts=mychain.jumptypes
plotly()
BAMF.histogram2D(mychain.states,sz,zoom,noisyroi,datastate)
