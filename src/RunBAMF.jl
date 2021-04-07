#Demonstrates use of BAMF on a single region
using Revise
using Plots
using ImageView  #comment out for debugging


## 
include("RJMCMC.jl")
includet("BAMF.jl")
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

n=2
sz=Int32(16)
sigma=1.3f0
x=[sz/2,sz/2-1]
y=[sz/2,sz/2+1]
photons=[777,500]
bg=1f-4

n=3
sz=Int32(16)
sigma=1.3f0
x=[sz/2,sz/2-1,sz/2+1]
y=[sz/2,sz/2+1,sz/2+1]
photons=[777,500,800]
bg=1f-4


datastate=BAMF.StateFlatBg(n,x,y,photons,bg)


# create a model
roi=BAMF.ArrayDD(sz)
BAMF.genmodel_2Dgauss!(datastate,sz,sigma,roi.data)
noisyroi=BAMF.poissrnd(roi)
#imshow(noisyroi.data)


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
burnin=10
acceptfuns=[BAMF.accept_move]
propfuns=[BAMF.propose_move]
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)

#create an intial state
state1=BAMF.calcintialstate(myRJ)

#run chain
#ImageView.closeall()
@time begin
mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1);
mychain=RJMCMC.buildchain(myRJMCMC,myRJ,datastate);
end

## Display
zoom=Int32(20)
jts=mychain.jumptypes
gr()
p=plot(jts,jts)
p2=histogram(jts)
p3=histogram(log.(mychain.α))
p4=histogram(mychain.accept)

plotly()
BAMF.histogram2D(mychain.states,sz,zoom)
BAMF.histogram2D(mychain.states,sz,zoom,noisyroi,datastate)






