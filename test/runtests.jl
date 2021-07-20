using BayesMultiFit
using Test
BAMF=BayesMultiFit
using ReversibleJumpMCMC
const RJMCMC = ReversibleJumpMCMC

#=
gendatastate(seed::Int32=-1, psf::PSF=PSF_airy2D(.2*pi))

Takes an optional seed, generates a 32 pixel image of 6 emitters with a 
mean photon emission of 1000, over 1000 iterations with a burn in of 1000
Most of the generation data is similar to that of the imagingsliver example.

Also takes an optional PSF, otherwise uses a PSF_airy2D with a v of .2*pi 
=# 

function psfget(psf::BAMF.PSF_airy2D)
    return psf.ν
end 

function psfget(psf::BAMF.PSF_gauss2D)
    return psf.σ
end

function gendatastate(seed::Int32=-1, psf::BAMF.PSF=BAMF.PSF_airy2D(.2*pi))
    if seed == -1
        Random.seed!()
    else
        Random.seed!(seed) 
    end
    
    # simulation config
    n=Int32(6)  #number of emitters
    μ=1000      #mean photons per emitter               
    iterations=1000
    burnin=1000
    sz=Int32(32)
    
    σ=.42f0*pi/psfget(psf)
    # setup prior distribution on intensity
    α=Float32(4)
    θ=Float32(μ/α)
    g=Gamma(α,θ)
    len=Int32(1024)
    θ_start=Float32(1)
    θ_step=Float32(5.0)
    pdf_x=range(θ_start,step=θ_step,length=len)
    mypdf=pdf(g,pdf_x)
    prior_photons=BAMF.RJPrior(len,θ_start,θ_step,mypdf)
    
    # set emitter positions and intensity
    x=sz/2f0*ones(Float32,n)+2*σ*randn(Float32,n)
    y=sz/2f0*ones(Float32,n)+ 2*σ*randn(Float32,n)
    photons=Float32.(rand(g,n))
    bg=1f-6
    datastate=BAMF.StateFlatBg(n,x,y,photons,bg)
    
    ## create a BAMF-type RJMCMC structure
    xystd=σ/10
    istd=10f0
    split_std=σ/2
    bndpixels=-20f0
    
    return datastate, psf,xystd,istd,split_std,data,bndpixels,prior_photons
end

function genBAMFDD(T::Type{BAMF.BAMFData}, sz::Int32=32)
    return genBAMFDD(T, sz)
end

function genBAMFDD(T::Type{BAMF.ArrayDD}, sz::Int32)
    return BAMF.ArrayDD(sz)
end

function genBAMFDD(T::Type{BAMF.DataSLIVER}, sz::Int32)
    return BAMF.DataSLIVER(sz, [1])
end

function DDinfo(DD::BAMF.BAMFData)
    dim= size(DD.data)
    size= DD.sz
    return size, dim, DD
end

function genBAMFSLIVER(T::Type{BAMF.BAMFData}, sz::Int32=32)
    return genBAMFSLIVER(T, sz)
end

function genBAMFSLIVER(T::Type{BAMF.DataSLIVER}, sz::Int32)
    return BAMF.DataSLIVER(sz, [2])
end

function SLIVERinfo(SLIVER::BAMF.DataSLIVER)
    dim= size(SLIVER.data)
    size= SLIVER.sz
    nimages= SLIVER.nimages
    inttime, invx, invy= SLIVER.inttime[1], SLIVER.invx[1], SLIVER.invy[1]
    return size, dim, nimages, inttime, invx, invy, SLIVER
end


# Create synthetic data
data=BAMF.ArrayDD(sz)      
BAMF.genmodel!(datastate,psf,data)
BAMF.poissrnd!(data.data)
# imshow(data.data)  #look at data  

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
    # test blank BAMFData object generation for DD type data for all BAMFData types
    DDdatatypelist=[BAMF.ArrayDD, BAMF.DataSLIVER]
    for datatype in DDdatatypelist
        data= genBAMFDD(datatype)
        sz, dim, DD=DDInfo(data)
        @test sz == 32
        @test dim == 32, 32
        @test isa(DD, datatype)
    end
    
    # test genBAMFData copy for DD type data for all BAMFData types
    for datatype in DDdatatypelist
        data= genBAMFDD(datatype)
        copy= BAMF.genBAMFData(data)
        sz, dim, DD=DDInfo(copy)
        @test sz == 32
        @test dim == 32, 32
        @test isa(DD, datatype)
    end
    
    datastate, psf,xystd,istd,split_std,data,bndpixels,prior_photons= gendatastate(Int32(1))
    
    # test genmodel! for DD type data for all BAMFData types
    expectDD=BAMF.ArrayDD(Int32(32))
    rjsDD = BAMF.RJStruct(32,psf,xystd,istd,split_std,expectDD,bndpixels,prior_photons)
    BAMF.genmodel!(datastate, rjsDD, expectDD)
    for datatype in DDdatatypelist
        data=genBAMFDD(datatype)
        BAMF.genmodel!(datastate, rjsDD, data)
        sz, dim, DD=DDInfo(data)
        @test sz == 32
        @test dim == 32, 32
        @test isa(DD, datatype)
        for i in 1:32, j in 1:32
            @test data.data[i, j] == expectDD.data[i, j]
        end
    end
    
    # test blank BAMFData object generation for SLIVER type data for all applicable BAMFData types
    SLIVERdatatypelist=[BAMF.DataSLIVER]
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        sz, dim, nimages, inttime, invx, invy, SLIVER = SLIVERInfo(data)
        @test sz == 32
        @test dim == 32, 32, 2
        @test nimages == 2
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
    end
    
    # test genBAMFData copy for SLIVER type data for all applicable BAMFData types
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        copy = BAMF.genBAMFData(data)
        sz, dim, nimages, inttime, invx, invy, SLIVER = SLIVERInfo(copy)
        @test sz == 32
        @test dim == 32, 32, 2
        @test nimages == 2
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
    end
    
    # test genmodel! for SLIVER type data for all applicable BAMFData types
    expectSLIVER=BAMF.DataSLIVER(32, [2])
    rjsSLIVER = BAMF.RJStruct(32,psf,xystd,istd,split_std,expectSLIVER,bndpixels,prior_photons)
    BAMF.genmodel!(datastate, rjsSLIVER, expectSLIVER)
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        BAMF.genmodel!(datastate, rjsSLIVER, data)
        sz, dim, nimages, inttime, invx, invy, SLIVER = SLIVERInfo(data)
        @test sz == 32
        @test dim == 32, 32, 2
        @test nimages == 2
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
        for i in 1:32, j in 1:32, k in 1:2
            @test data.data[i, j, k] == expectSLIVER.data[i, j, k]
        end 
    end
    
    #=test deterministic seed for calcintialstate and buildchain 
    
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
    @test length(n_states) == 4869=#
    
end
