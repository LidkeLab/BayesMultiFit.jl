## An example of emitter estimation from a combined direct detection and SLIVER measurement

import DisplayBAMF
Disp=DisplayBAMF
using ReversibleJumpMCMC
const RJMCMC = ReversibleJumpMCMC
using Plots
using Distributions
using MATLAB
#ImageView.closeall()

# simulation config
n=Int32(5)  #number of emitters
μ=1000      #mean photons per emitter               
iterations=1000
burnin=1000

# telescope parameters
f=2.0f0
D=0.1f0
λ=1f-6 

# detector parameters
zoom=1f0
pixelsize=4f-6*zoom
sz=Int32(32/zoom)

# setup the psf 
ν=π*D/(λ*f)*pixelsize
psf=BAMF.PSF_airy2D(ν)
σ=.42f0*pi/ν  #used for mcmc jump size calculations

# create an empty dataset structure
# ?BAMF.DataSLIVER 

nmeas=2
if nmeas==1
    type=Int32[1] #DD
end
if nmeas==2
    type=Int32[1,2] #DD then SLIVER
end

data=BAMF.DataSLIVER(sz,type)

# setup prior distribution on intensity
using Distributions
α=Float32(4)
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
x=sz/2f0*ones(Float32,n)+2*σ*randn(Float32,n)
y=sz/2f0*ones(Float32,n)+ 2*σ*randn(Float32,n)
photons=Float32.(rand(g,n))
bg=1f-6
datastate=BAMF.StateFlatBg(n,x,y,photons,bg)

# set inversion points
data.invx=sz/2f0*ones(Float32,nmeas)
data.invy=sz/2f0*ones(Float32,nmeas)

# generate the data 
BAMF.genmodel!(datastate,sz,psf,data)

## Profiling and Timing
#  using BenchmarkTools
#  @benchmark BAMF.genmodel!(datastate,sz,psf,data)
#  @time BAMF.genmodel!(datastate,sz,psf,data)
# using ProfileView
# ProfileView.@profview BAMF.genmodel!(datastate,sz,psf,data) # run once to trigger compilation (ignore this one)
# ProfileView.@profview BAMF.genmodel!(datastate,sz,psf,data)

# make data noisy 
BAMF.poissrnd!(data.data)
# imshow(data.data)


## create a BAMF-type RJMCMC structure
xystd=σ/10
istd=10f0
split_std=σ/2
bndpixels=-20f0
myRJ=BAMF.RJStruct(sz,psf,xystd,istd,split_std,data,bndpixels,prior_photons)

## setup the RJMCMC.jl model
# Jumptypes are: move, bg, add, remove, split, merge
njumptypes=6
jumpprobability=[1,0,.1,.1,.1,.1] 
jumpprobability=jumpprobability/sum(jumpprobability)

# create an RJMCMC structure with all model info
acceptfuns=[BAMF.accept_move,BAMF.accept_bg,BAMF.accept_add,BAMF.accept_remove,BAMF.accept_split,BAMF.accept_merge] #array of functions
propfuns=[BAMF.propose_move,BAMF.propose_bg,BAMF.propose_add,BAMF.propose_remove,BAMF.propose_split,BAMF.propose_merge] #array of functions
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)

#create an intial state
state1=BAMF.calcintialstate(myRJ)

## run chain
@time mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1);
# @time mychain=RJMCMC.buildchain(myRJMCMC,myRJ,datastate)



## Profiling
# using ProfileView
# ProfileView.@profview  mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1) # run once to trigger compilation (ignore this one)
# ProfileView.@profview mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1)

## Display
# accepts,pl2=RJMCMC.showacceptratio(mychain)

zm=Int32(zoom)
plotly()
plt=Disp.histogram2D(mychain.states,sz,zm,datastate)
display(plt)

map_n,posterior_n,traj_n=BAMF.getn(mychain.states)
plt2=plot(traj_n)
display(plt2)
# BAMF.showoverlay(mychain.states,myRJ)
postimage=BAMF.getposterior(mychain.states,sz,Int32(4))
heatmap(postimage)

## MAPN Results
states_mapn,n=BAMF.getmapnstates(mychain.states)
plt=Disp.histogram2D(states_mapn,sz,zm,datastate)
display(plt)
Results_mapn=BAMF.getmapn(mychain.states)
Disp.plotstate(datastate,Results_mapn)


## Testing the matlab interface


out=BAMF.matlab_SLIVER_FlatBG(data.data,data.type,data.invx,data.invy,"airy",ν,θ_start,θ_step,len,mypdf,Int32(burnin),Int32(iterations),xystd,istd,split_std,bndpixels)

# mex interface

args=[MATLAB.mxarray(data.data),MATLAB.mxarray(data.type),MATLAB.mxarray(data.invx),MATLAB.mxarray(data.invy),
MATLAB.mxarray("airy"),MATLAB.mxarray(ν),MATLAB.mxarray(θ_start),MATLAB.mxarray(θ_step),
MATLAB.mxarray(len),MATLAB.mxarray(mypdf),MATLAB.mxarray(Int32(burnin)),MATLAB.mxarray(Int32(iterations)),
MATLAB.mxarray(xystd),MATLAB.mxarray(istd),MATLAB.mxarray(split_std),MATLAB.mxarray(bndpixels)
]

BAMF.mextypes(args)
BAMF.mextest(args)

mapn=BAMF.matlab_SLIVER_FlatBG_mex(args)










