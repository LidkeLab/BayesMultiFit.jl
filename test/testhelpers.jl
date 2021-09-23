#= 
"""
    psfget(psf::BAMF.PSF)

psfget returns the contents of a PSF regardless of type
"""
function psfget(psf::BAMF.PSF_airy2D)
    return psf.ν
end 

function psfget(psf::BAMF.PSF_gauss2D)
    return psf.σ
end
 =#
"""
    gendatastate(;seed::Integer=-1, psf::PSF=PSF_airy2D(.2*pi), n::Int32=6)

    Takes an optional Integer seed, set to random if not opted for
    An optional Int32 number of emitters n, defaulted to 6
    optional integer mean emitters per photon μ, defaulted to 1000
    optional Int32 pixel number sz, defaulted to 32
    optional Float32 intensity distribution shape parameter α, defaulted to 4f0
    optional Float32 background bg, defaulted to 1f-6
    Also takes an optional PSF, otherwise uses a PSF_airy2D with a v of .2*pi 
    optional Float32 intensity standard deviation istd, defaulted to 10f0
    optional Float32 bndpixels, defaulted to -20f0

Outputs a StateFlatBg datastate, as well as all the information needed for an RJStruct sans the BAMFData object as a tuple, in the following order: datastate, psf, sz, xystd, istd, split_std, bndpixels, prior_photons
"""
function gendatastate(seed::Int32=Int32(-1); psf::PSF.PSF=PSF.Airy2D(1.2,0.6,0.1), n::Int32=Int32(6), μ::Int64=1000, sz::Int32=Int32(32), α::Float32=Float32(4), bg::Float32=1f-6, istd::Float32=10f0, bndpixels::Float32=-20f0)
    if seed == -1
        Random.seed!()
    else
        Random.seed!(seed) 
    end
    
    
    σ=.42f0*pi/psf.ν
    # setup prior distribution on intensity
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
    datastate=BAMF.StateFlatBg(Int64(n),convert(Array{Float32},x),convert(Array{Float32},y),photons,bg)
    
    ## create a BAMF-type RJMCMC structure
    xystd=σ/10
    split_std=σ/2
    
    return datastate, psf, xystd, istd, split_std, bndpixels, prior_photons
end

"""
    genBAMFDD(T::Type{BAMF.BAMFData}, sz::Int32) 

generates an empty BAMFData direct detection model of type T and size sz. The default size is 32.
"""
function genBAMFDD(T::Type{BAMF.ArrayDD}, sz::Int32=Int32(32))
    return BAMF.ArrayDD(sz)
end

#= function genBAMFDD(T::Type{BAMF.DataSLIVER}, sz::Int32=Int32(32))
    return BAMF.DataSLIVER(sz, [Int32(1)])
end

function genBAMFDD(T::Type{BAMF.AdaptData}, sz::Int32=Int32(32))
    return BAMF.AdaptData(sz, [(BAMF.DDMeasType, ())])
end =#

"""
    DDinfo(DD::BAMF.BAMFData) 

returns a tuple of the information that ought to be contained in a direct detection BAMFData structure: size, an array of the appropriate dimensions, and the object itself.
"""
function DDinfo(DD::BAMF.BAMFData)
    dim= size(DD.data)
    sz= DD.sz
    return sz, dim, DD
end
#= 
"""
    genBAMFSLIVER(T::Type{BAMF.BAMFData}, sz::Int32) 

returns an empty BAMFData SLIVER model of type T and size sz. The default size is 32.
"""
function genBAMFSLIVER(T::Type{BAMF.DataSLIVER}, sz::Int32=Int32(32))
    return BAMF.DataSLIVER(sz, [Int32(2)])
end

function genBAMFSLIVER(T::Type{BAMF.AdaptData}, sz::Int32=Int32(32))
    return BAMF.AdaptData(sz, [(BAMF.SLIVERMeasType, (Float32(0f0), Float32(0f0)))])
end

"""
    SLIVERinfo(SLIVER::BAMF.BAMFData) 
    
returns a tuple of the information that ought to be contained in a SLIVER BAMFData structure: size, an array of the appropriate dimensions, the object itself, and the x y inversion points alongside the integration time.
"""
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
 =#
"""
    genRJMCMC(burnin::Int32, iterations::Int32) 

a function that produces an RJMCMCStruct with the usual DD accept and propose functions alongside some standard jump probabilities. The default burnin and iterations is 1000.
"""
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
    return myRJMCMC
end