using BayesMultiFit
using Test
BAMF=BayesMultiFit
using ReversibleJumpMCMC
const RJMCMC = ReversibleJumpMCMC
using Random
using Distributions

#=
psfget returns the contents of a PSF regardless of type
=#

function psfget(psf::BAMF.PSF_airy2D)
    return psf.ν
end 

function psfget(psf::BAMF.PSF_gauss2D)
    return psf.σ
end

#=
gendatastate(seed::Int32=-1, psf::PSF=PSF_airy2D(.2*pi))

Takes an optional seed, generates a 32 pixel image of 6 emitters with a 
mean photon emission of 1000, over 1000 iterations with a burn in of 1000
Most of the generation data is similar to that of the imagingsliver example.

Also takes an optional PSF, otherwise uses a PSF_airy2D with a v of .2*pi 
=# 

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
    
    return datastate, psf,xystd,istd,split_std,bndpixels,prior_photons
end

#=
genBAMFDD(T::Type{BAMF.BAMFData}, sz::Int32) generates an empty BAMFData direct detection model of 
type T and size sz. The default size is 32.
=#

function genBAMFDD(T::Type{BAMF.ArrayDD}, sz::Int32=Int32(32))
    return BAMF.ArrayDD(sz)
end

function genBAMFDD(T::Type{BAMF.DataSLIVER}, sz::Int32=Int32(32))
    return BAMF.DataSLIVER(sz, [Int32(1)])
end

function genBAMFDD(T::Type{BAMF.AdaptData}, sz::Int32=Int32(32))
    return BAMF.AdaptData(sz, [(BAMF.DDMeasType, ())])
end

#=
DDinfo(DD::BAMF.BAMFData) returns a tuple of the information that ought to be contained in a direct
detection BAMFData structure: size, an array of the appropriate dimensions, and the object itself.
=#

function DDinfo(DD::BAMF.BAMFData)
    dim= size(DD.data)
    sz= DD.sz
    return sz, dim, DD
end

#=
genBAMFSLIVER(T::Type{BAMF.BAMFData}, sz::Int32) returns an empty BAMFData SLIVER model of type T and
size sz. The default size is 32.
=#

function genBAMFSLIVER(T::Type{BAMF.DataSLIVER}, sz::Int32=Int32(32))
    return BAMF.DataSLIVER(sz, [Int32(2)])
end

function genBAMFSLIVER(T::Type{BAMF.AdaptData}, sz::Int32=Int32(32))
    return BAMF.AdaptData(sz, [(BAMF.SLIVERMeasType, (Float32(0f0), Float32(0f0)))])
end

#=
SLIVERinfo(SLIVER::BAMF.BAMFData) returns a tuple of the information that ought to be contained in a SLIVER
BAMFData structure: size, an array of the appropriate dimensions, the object itself, and the x y inversion
points alongside the integration time.
=#

function SLIVERinfo(SLIVER::BAMF.DataSLIVER)
    dim= size(SLIVER.data)
    sz= SLIVER.sz
    inttime, invx, invy= SLIVER.inttime[1], SLIVER.invx[1], SLIVER.invy[1]
    return sz, dim, inttime, invx, invy, SLIVER
end

function SLIVERinfo(SLIVER::BAMF.AdaptData)
    dim= size(SLIVER.data)
    sz= SLIVER.sz
    inttime, invx, invy= SLIVER.meastypes[1].inttime, SLIVER.meastypes[1].invx, SLIVER.meastypes[1].invy
    return sz, dim, inttime, invx, invy, SLIVER
end

#=
genRJMCMC(burnin::Int32, iterations::Int32) is a function that produces an RJMCMCStruct with the usual DD accept 
and propose functions alongside some standard jump probabilities. The default burnin and iterations is 1000.
=#

function genRJMCMC(burnin::Int32=Int32(1000), iterations::Int32=Int32(1000))
    ## setup the RJMCMC.jl model
    # Jumptypes are: move, bg, add, remove, split, merge
    njumptypes=6
    jumpprobability=[1,0,.1,.1,.1,.1] # Model with no bg 
    jumpprobability=jumpprobability/sum(jumpprobability)

    # create an RJMCMC structure with all model info
    acceptfuns=[BAMF.accept_move,BAMF.accept_bg,BAMF.accept_add,BAMF.accept_remove,BAMF.accept_split,BAMF.accept_merge] #array of functions
    propfuns=[BAMF.propose_move,BAMF.propose_bg,BAMF.propose_add,BAMF.propose_remove,BAMF.propose_split,BAMF.propose_merge] #array of functions
    myRJMCMC=RJMCMC.RJMCMCStruct(burnin,iterations,njumptypes,jumpprobability,propfuns,acceptfuns)
end
#=
# Create synthetic data
data=BAMF.ArrayDD(sz)      
BAMF.genmodel!(datastate,psf,data)
BAMF.poissrnd!(data.data)
# imshow(data.data)  #look at data  

=#

@testset "BayesMultiFit.jl" begin
    # test blank BAMFData object generation for DD type data for all BAMFData types
    DDdatatypelist=[BAMF.ArrayDD, BAMF.DataSLIVER, BAMF.AdaptData]
    for datatype in DDdatatypelist
        data= genBAMFDD(datatype)
        sz, dim, DD=DDinfo(data)
        @test sz == 32
        if length(dim)==3
            @test dim==(32, 32, 1)
        else
            @test dim==(32,32)
        end
        @test isa(DD, datatype)
    end
    
    # test genBAMFData copy for DD type data for all BAMFData types
    for datatype in DDdatatypelist
        data= genBAMFDD(datatype)
        copy= BAMF.genBAMFData(data)
        sz, dim, DD=DDinfo(copy)
        @test sz == 32
        if length(dim)==3
            @test dim==(32, 32, 1)
        else
            @test dim==(32,32)
        end
        @test isa(DD, datatype)
    end
    
    datastate, psf,xystd,istd,split_std,bndpixels,prior_photons= gendatastate(Int32(1))
    
    # test genmodel! for DD type data for all BAMFData types
    expectDD=BAMF.ArrayDD(Int32(32))
    rjsDD = BAMF.RJStruct(32,psf,xystd,istd,split_std,expectDD,bndpixels,prior_photons)
    BAMF.genmodel!(datastate, rjsDD, expectDD)
    for datatype in DDdatatypelist
        data=genBAMFDD(datatype)
        BAMF.genmodel!(datastate, rjsDD, data)
        sz, dim, DD=DDinfo(data)
        @test sz == 32
        if length(dim)==3
            @test dim==(32, 32, 1)
            for i in 1:32, j in 1:32
                @test data.data[i, j, 1] ≈ expectDD.data[i, j] atol=1f-4
            end
        else
            @test dim==(32,32)
            for i in 1:32, j in 1:32
                @test data.data[i, j] ≈ expectDD.data[i, j] atol=1f-4
            end
        end
        @test isa(DD, datatype)
        
    end
    
    # test blank BAMFData object generation for SLIVER type data for all applicable BAMFData types
    SLIVERdatatypelist=[BAMF.DataSLIVER, BAMF.AdaptData]
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        sz, dim, inttime, invx, invy, SLIVER = SLIVERinfo(data)
        @test sz == 32
        @test dim == (32, 32, 2)
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
    end
    
    # test genBAMFData copy for SLIVER type data for all applicable BAMFData types
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        copy = BAMF.genBAMFData(data)
        sz, dim, inttime, invx, invy, SLIVER = SLIVERinfo(copy)
        @test sz == 32
        @test dim == (32, 32, 2)
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
    end
    
    # test genmodel! for SLIVER type data for all applicable BAMFData types
    expectSLIVER=BAMF.DataSLIVER(Int32(32), [Int32(2)])
    rjsSLIVER = BAMF.RJStruct(Int32(32),psf,xystd,istd,split_std,expectSLIVER,bndpixels,prior_photons)
    BAMF.genmodel!(datastate, rjsSLIVER, expectSLIVER)
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        BAMF.genmodel!(datastate, rjsSLIVER, data)
        sz, dim, inttime, invx, invy, SLIVER = SLIVERinfo(data)
        @test sz == 32
        @test dim == (32, 32, 2)
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
        for i in 1:32, j in 1:32, k in 1:2
            @test data.data[i, j, k] ≈ expectSLIVER.data[i, j, k] atol=1f-4
        end 
    end
    
    # test deterministic seed for calcintialstate and buildchain 
    datastate, psf,xystd,istd,split_std,bndpixels,prior_photons= gendatastate(Int32(1))
    
    expectDD=BAMF.ArrayDD(Int32(32))
    rjsDD = BAMF.RJStruct(32,psf,xystd,istd,split_std,expectDD,bndpixels,prior_photons)
    BAMF.genmodel!(datastate, rjsDD, expectDD)
    stateDD = BAMF.calcintialstate(rjsDD, Int32(5));
    RJMCMCDD = genRJMCMC();
    
    mychain=RJMCMC.buildchain(RJMCMCDD,rjsDD,stateDD);
    map_n,posterior_n,traj_n=BAMF.getn(mychain.states);
    exp_posterior = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.032546673, 0.054948926, 0.22437479, 0.2593871, 0.14054245, 0.21754138, 0.070658684];
    n_states, map_n=BAMF.getmapnstates(mychain.states);
    @test map_n == 14
    @test length(posterior_n) == 18
    for i in 1:18
        @test posterior_n[i] ≈ exp_posterior[i] atol=0.0000001
    end
    @test length(n_states) == 263
end
