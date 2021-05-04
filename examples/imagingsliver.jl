## An example of emitter estimation from a combined direct detection and SLIVER measurement

include("../src/RJMCMC.jl")
include("../src/BAMF.jl")
using ImageView
using Plots
ImageView.closeall()

# simulation config
n=Int32(5)  #number of emitters
μ=1000      #mean photons per emitter               

# telescope parameters
f=2.0f0
D=0.1f0
λ=1f-6 

# detector parameters
pixelsize=1f-6
sz=Int32(128)

# setup the psf 
ν=π*D/(λ*f)*pixelsize
psf=BAMF.PSF_airy2D(ν)
σ=.42f0*pi/ν  #used for mcmc jump size calculations

# create an empty dataset structure
# ?BAMF.DataSLIVER 

# type=Int32[1] #DD then SLIVER
type=Int32[1,2] #DD then SLIVER
data=BAMF.DataSLIVER(sz,type)

# setup prior distribution on intensity
using Distributions
α=Float32(2)
θ=Float32(μ/α)
g=Gamma(α,θ)
len=Int32(1024)
θ_start=Float32(1)
θ_step=Float32(5.0)
pdf_x=range(θ_start,step=θ_step,length=len)
mypdf=pdf(g,pdf_x)
plt=plot(pdf_x,mypdf)
display(plt)
prior_photons=BAMF.RJPrior(len,θ_start,θ_step,mypdf)

# set emitter positions and intensity
x=sz/2f0*ones(Float32,n)+σ*randn(Float32,n)
y=sz/2f0*ones(Float32,n)+ σ*randn(Float32,n)
photons=Float32.(rand(g,n))
bg=1f-6
datastate=BAMF.StateFlatBg(n,x,y,photons,bg)

# set inversion points
data.invx=sz/2f0*ones(Float32,2)
data.invy=sz/2f0*ones(Float32,2)

# generate the data 
BAMF.genmodel!(datastate,sz,psf,data)
sum(data.data)
sum(datastate.photons)
# imshow(data.data)

# make data noisy 
BAMF.poissrnd!(data.data)

## create a BAMF-type RJMCMC structure

xystd=σ/20
istd=10f0
split_std=σ/2
bndpixels=0f0
myRJ=BAMF.RJStruct(sz,psf,xystd,istd,split_std,data,bndpixels,prior_photons)

data2=BAMF.genBAMFData(myRJ)
# BAMF.genmodel!(datastate,myRJ,data2)
# BAMF.genmodel!(datastate,myRJ.sz,myRJ.psf,data2)
# imshow(data2.data)

## setup the RJMCMC.jl model
#Jumptypes are: move, bg, add, remove, split, merge
njumptypes=6
jumpprobability=[1,0,.1,.1,.1,.1] #Move only
jumpprobability=jumpprobability/sum(jumpprobability)

# create an RJMCMC structure with all model info
iterations=500
burnin=10
acceptfuns=[BAMF.accept_move,BAMF.accept_bg,BAMF.accept_add,BAMF.accept_remove,BAMF.accept_split,BAMF.accept_merge] #array of functions
propfuns=[BAMF.propose_move,BAMF.propose_bg,BAMF.propose_add,BAMF.propose_remove,BAMF.propose_split,BAMF.propose_merge] #array of functions
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)

#create an intial state
state1=BAMF.calcintialstate(myRJ)

## run chain
@time mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1)

## Profiling
using ProfileView
ProfileView.@profview  mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1) # run once to trigger compilation (ignore this one)
ProfileView.@profview mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1)

## Display
zm=Int32(1)
jts=mychain.jumptypes
plotly()
# BAMF.histogram2D(mychain.states,sz,zm,data,datastate)
plt=BAMF.histogram2D(mychain.states,sz,zm,datastate)
display(plt)

map_n,posterior_n,traj_n=BAMF.getn(mychain.states)
plot(traj_n)

# println(map_n)
# println(posterior_n)
sum(mychain.states[end].photons)
sum(datastate.photons)
BAMF.showoverlay(mychain.states,myRJ)
accepts,pl2=RJMCMC.showacceptratio(mychain)
# plotly()
# BAMF.plotstate(datastate,mychain.states[end])








