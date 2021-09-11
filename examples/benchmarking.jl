

using Revise
using BayesMultiFit
BAMF=BayesMultiFit
using ReversibleJumpMCMC
const RJMCMC = ReversibleJumpMCMC
using Plots
using Distributions
using MicroscopePSFs
const PSF=MicroscopePSFs
using ProfileView
using BenchmarkTools


## simulation config
n=6             # number of emitters
μ=1000          # mean photons per emitter               
sz=16           # ROI size in pixels
iterations=1000 # RJMCMC iterations
burnin=1000     # RJMCMC iterations for burn-in

## PSF config
σ=1.3f0 # Gaussian PSF Sigma in Pixels
pixelsize=1.0
psf=PSF.Gauss2D(σ,pixelsize)
 psf=PSF.InterpolatedPSF(psf,(sz*2,sz*2))

## setup prior distribution on intensity
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
x=sz*rand(Float32,n)
y=sz*rand(Float32,n)

photons=Float32.(rand(g,n))
bg=1f-6
datastate=BAMF.StateFlatBg(n,x,y,photons,bg)

## Create synthetic data
data=BAMF.ArrayDD(sz)      

@code_warntype BAMF.genmodel!(datastate,psf,data)
@btime BAMF.genmodel!(datastate,psf,data)
@benchmark BAMF.genmodel!(datastate,psf,data)

roi=[(i,j) for i=1:sz,j=1:sz]
@benchmark MicroscopePSFs.pdf(psf,(0,0),(0,0))
@benchmark MicroscopePSFs.pdf(psf,roi,(0,0))

## Create synthetic data
data=BAMF.ArrayDD(sz)      
BAMF.genmodel!(datastate,psf,data)
BAMF.poissrnd!(data.data)
# imshow(data.data)  #look at data  

## create a BAMF-type RJMCMC structure
xystd=σ/10
istd=10f0
split_std=σ/2
bndpixels=0f0
myRJ=BAMF.RJStruct(sz,psf,xystd,istd,split_std,data,bndpixels,prior_photons,BAMF.ArrayDD(sz),BAMF.ArrayDD(sz))

## setup the RJMCMC.jl model
# Jumptypes are: move, bg, add, remove, split, merge
njumptypes=6
jumpprobability=[1,0,.1,.1,.1,.1] # Model with no bg 
jumpprobability=jumpprobability/sum(jumpprobability)

# create an RJMCMC structure with all model info
acceptfuns=[BAMF.accept_move,BAMF.accept_bg,BAMF.accept_add,BAMF.accept_remove,BAMF.accept_split,BAMF.accept_merge] #array of functions
propfuns=[BAMF.propose_move,BAMF.propose_bg,BAMF.propose_add,BAMF.propose_remove,BAMF.propose_split,BAMF.propose_merge] #array of functions
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)
#create an intial state
state1=BAMF.calcintialstate(myRJ)


## run chain. This is the call to the main algorithm
@code_warntype RJMCMC.buildchain(myRJMCMC,myRJ,state1)
ProfileView.@profview RJMCMC.buildchain(myRJMCMC,myRJ,state1)
@benchmark mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1)






