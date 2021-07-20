using BayesMultiFit
using Test
BAMF=BayesMultiFit
using ReversibleJumpMCMC
const RJMCMC = ReversibleJumpMCMC

## simulation config
n=Int32(6)      # number of emitters
μ=1000          # mean photons per emitter               
sz=Int32(16)    # ROI size in pixels
iterations=5000 # RJMCMC iterations
burnin=5000     # RJMCMC iterations for burn-in

## PSF config
σ=1.3f0 # Gaussian PSF Sigma in Pixels
psf=BAMF.PSF_gauss2D(σ)

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

# set emitter positions and intensity with a random seed; same positions and data produced each time
Random.seed!(1);
x=sz/2f0*ones(Float32,n)+3*σ*randn(Float32,n)
y=sz/2f0*ones(Float32,n)+ 3*σ*randn(Float32,n)
photons=Float32.(rand(g,n))
bg=1f-6
datastate=BAMF.StateFlatBg(n,x,y,photons,bg)

## Create synthetic data
data=BAMF.ArrayDD(sz)      
BAMF.genmodel!(datastate,psf,data)
BAMF.poissrnd!(data.data)
# imshow(data.data)  #look at data  

## create a BAMF-type RJMCMC structure
xystd=σ/10
istd=10f0
split_std=σ
bndpixels=0f0
myRJ=BAMF.RJStruct(sz,psf,xystd,istd,split_std,data,bndpixels,prior_photons)

## setup the RJMCMC.jl model
# Jumptypes are: move, bg, add, remove, split, merge
njumptypes=6
jumpprobability=[1,0,.1,.1,.1,.1] # Model with no bg 
jumpprobability=jumpprobability/sum(jumpprobability)

# create an RJMCMC structure with all model info
acceptfuns=[BAMF.accept_move,BAMF.accept_bg,BAMF.accept_add,BAMF.accept_remove,BAMF.accept_split,BAMF.accept_merge] #array of functions
propfuns=[BAMF.propose_move,BAMF.propose_bg,BAMF.propose_add,BAMF.propose_remove,BAMF.propose_split,BAMF.propose_merge] #array of functions
myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)

@testset "BayesMultiFit.jl" begin
    # Write your tests here.
    
    state1 = BAMF.calcintialstate(myRJ, 5);
    mychain=RJMCMC.buildchain(myRJMCMC,myRJ,state1);
    map_n,posterior_n,traj_n=BAMF.getn(mychain.states);
    exp_posterior = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9695662, 0.030433772];
    n_states, map_n=BAMF.getmapnstates(mychain.states);
    @test map_n == 6
    @test length(posterior_n) == 8
    for i in 1:8
        @test posterior_n[i] ≈ exp_posterior[i] atol=0.0000001
    end
    @test length(n_states) == 4869
    
end
