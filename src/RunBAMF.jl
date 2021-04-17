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


##  create a model
dataroi=BAMF.ArrayDD(sz)
BAMF.genmodel_2Dgauss!(datastate,sz,sigma,dataroi.data)
noisyroi=BAMF.poissrnd(dataroi)
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
iterations=5000
burnin=1000
acceptfuns=[BAMF.accept_move]
propfuns=[BAMF.propose_move]
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)

#create an intial state
state1=BAMF.calcintialstate(myRJ)

## run chain
#ImageView.closeall()
using Profile
using ProfileView
ProfileView.@profview mychain=RJMCMC.buildchain(myRJMCMC,myRJ,datastate)

@time mychain=RJMCMC.buildchain(myRJMCMC,myRJ,datastate)



## Display

zoom=Int32(20)
jts=mychain.jumptypes
#gr()
#p3=histogram(log.(mychain.α))
#p4=histogram(mychain.accept)

plotly()
#BAMF.histogram2D(mychain.states,sz,zoom)
BAMF.histogram2D(mychain.states,sz,zoom,noisyroi,datastate)

## Testing cuda kernels
using CUDA

n=3
sz=Int32(16)
sigma=1.3f0
x=[sz/2,sz/2-1,sz/2+1]
y=[sz/2,sz/2+1,sz/2+1]
photons=[777,500,800]
bg=1f-4
s=BAMF.StateFlatBg(n,x,y,photons,bg)

σ=1.3f0
dataroi=BAMF.ArrayDD(sz)

function genmodel_2Dgauss_CUDA!(s_n::Int32,s_x, s_y, 
    s_photons,s_bg::Float32, sz::Int32, σ::Float32, model)
    #note that the 2d array is linearized and using 1-based indexing in kernel
    ii = blockIdx().x
    jj = threadIdx().x 
    
    idx=(ii-1)*sz+jj
    model[idx] = s_bg + 1f-4
    for nn = 1:s_n
        model[idx] += s_photons[nn] / (2 * π * σ^2) *
                exp(-(ii - s_y[nn])^2 / (2 * σ^2)) *
                exp(-(jj - s_x[nn])^2 / (2 * σ^2))
    end
    return nothing
end

CUDA.@time  begin
    for ii=1:1f4
        CUDA.@sync begin
        @cuda threads=sz blocks=sz genmodel_2Dgauss_CUDA!(s.n,s.x,s.y,s.photons,s.bg, sz, σ, dataroi.data)
        end
    end
end

 @cuda threads=sz blocks=sz BAMF.genmodel_2Dgauss_CUDA!(s.n,s.x,s.y,s.photons,s.bg, sz, σ, dataroi.data)


##
using CUDA

# C=CUDA.CuPrimaryContext(CUDA.device())
# CUDA.unsafe_reset!(C)

sz=16
dataroi=BAMF.ArrayDD(sz)
model=dataroi.data

function testkernel1(model)
    ii = blockIdx().x
    jj = threadIdx().x 
    model[ii]=5
    return nothing
end

@cuda threads=sz blocks=sz testkernel1(model)

##

using CUDA

# C=CUDA.CuPrimaryContext(CUDA.device())
# CUDA.unsafe_reset!(C)

sz=Int32(16)
dataroi=BAMF.ArrayDD(sz)
model=dataroi.data
LLR=CuArray{Float32}(undef,1)

ProfileView.@profview begin
    for ii=1:10000
@cuda threads=sz blocks=sz BAMF.likelihoodratio_CUDA!(sz,model,model,model,LLR)
    end
end


## random number generator
using GPUArrays
using CUDA


A = CuArray{Float32}(undef,1)
A[1]=1000

function kernel_rand!(state, randstate)
    random_number = GPUArrays.gpu_rand(Float32, state, randstate)
    CUDA.@cuprintln(random_number)
    return
end



@cuda kernel_rand!


gpu_call(kernel_rand, A, (randstate,))

