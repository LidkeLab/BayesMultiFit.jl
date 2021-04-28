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

## Scaling factor for testing
zoom=8;

## make prior on emitter intensity distributions
using Distributions
α=Float32(10)
θ=Float32(100)
g=Gamma(α,θ)
len=Int32(1024)
θ_start=Float32(1)
θ_step=Float32(2.0)
x=range(θ_start,step=θ_step,length=len)
mypdf=pdf(g,x)
plot(x,mypdf)
prior_photons=BAMF.RJPrior(len,θ_start,θ_step,mypdf)

## create a psf
σ=Float32(1.3*zoom)
psf=BAMF.PSF_gauss2D(σ)

## create the true state of a dataset 
n=Int32(3)
sz=Int32(16*zoom)
sigma=1.3f0*zoom
x=sz/2f0*ones(Float32,n) +zoom*randn(Float32,n)
y=sz/2f0*ones(Float32,n) +zoom*randn(Float32,n)
photons=Float32.(rand(g,n))
bg=1f-4
datastate=BAMF.StateFlatBg(n,x,y,photons,bg)

##  create a model then make noisy
dataroi=BAMF.ArrayDD(sz)
BAMF.genmodel!(datastate,sz,psf,dataroi)
noisyroi=BAMF.poissrnd(dataroi)


## create a BAMF-type RJMCMC structure
xystd=sigma/10;
istd=10;
split_std=sigma/2
bndpixels=2
myRJ=BAMF.RJStructDD(sz,psf,xystd,istd,split_std,noisyroi,bndpixels,prior_photons)

## setup the RJMCMC.jl model
#Jumptypes are: move, bg, add, remove, split, merge
njumptypes=6
jumpprobability=[1,0,.05,.05,.05,.05] #Move only
jumpprobability=jumpprobability/sum(jumpprobability)

# create an RJMCMC structure with all model info
iterations=5000
burnin=50
acceptfuns=[BAMF.accept_move,BAMF.accept_bg,BAMF.accept_add,BAMF.accept_remove,BAMF.accept_split,BAMF.accept_merge] #array of functions
propfuns=[BAMF.propose_move,BAMF.propose_bg,BAMF.propose_add,BAMF.propose_remove,BAMF.propose_split,BAMF.propose_merge] #array of functions
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)

#create an intial state
state1=BAMF.calcintialstate(myRJ)

## run chain
@time mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1)

## Display
zm=Int32(round(20/zoom))
jts=mychain.jumptypes
plotly()
BAMF.histogram2D(mychain.states,sz,zm,noisyroi,datastate)
BAMF.histogram2D(mychain.states,sz,zm,datastate)
plot(traj_n)

map_n,posterior_n,traj_n=BAMF.getn(mychain.states)
plot(traj_n)
println(map_n)
println(posterior_n)
sum(mychain.states[end].photons)
sum(datastate.photons)
