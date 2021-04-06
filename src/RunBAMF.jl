#Demonstrates use of BAMF on a single region
using Plots
#using ImageView

## 
include("RJMCMC.jl")
include("BAMF.jl")
using .RJMCMC

#Jumptypes are:
#move, bg, add,remove, split,merge



## Create a dataset 

# creating a state
n=1
sz=Int32(16)
sigma=1.3f0
x=[sz/2]
y=[sz/2]
photons=[777]
bg=1f-4
datastate=BAMF.StateFlatBg(n,x,y,photons,bg)
# create a model
noisyroi=BAMF.ArrayDD(sz)
BAMF.genmodel_2Dgauss!(datastate,sz,sigma,noisyroi.data)


## Create a BAMF RJ struct
xystd=sigma/10;
istd=10;
myRJ=BAMF.RJStructDD(sz,sigma,xystd,istd,noisyroi)

## Setup the RJMCMC.jl model

njumptypes=6
jumpprobability=[1,0,0,0,0,0] #Move only
jumpprobability=jumpprobability/sum(jumpprobability)

# create a structure with all model info
iterations=100
burnin=1
acceptfuns=[BAMF.accept_move]
propfuns=[BAMF.propose_move]
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)

#create an intial state
state1=BAMF.calcintialstate(myRJ)

#run chain
#ImageView.closeall()
mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1)

## Display



jts=mychain.jumptypes
histogram(jts)

histogram(mychain.α)
histogram(mychain.accept)


zoom=Int32(20);
BAMF.histogram2D(mychain.states,sz,zoom)




