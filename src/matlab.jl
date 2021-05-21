## This provides a mex-style interface for calling from MATLAB via Mex.jl

using MATLAB
using Printf

function mextest(args::Vector{MATLAB.MxArray})
    return [MATLAB.jvalue(arg) for arg in args]
end


function mextypes(args::Vector{MATLAB.MxArray})
    return [@sprintf("%s",typeof(MATLAB.jvalue(arg))) for arg in args]
end

function matlab_DD_FlatBG_mex_lite(args::Vector{MATLAB.MxArray})
    roi = MATLAB.jvalue(args[1])
    
    psftype = MATLAB.jvalue(args[2])
    σ_psf = MATLAB.jvalue(args[3])
    θ_start = MATLAB.jvalue(args[4])
    θ_step = MATLAB.jvalue(args[5])
    len = MATLAB.jvalue(args[6])
    pdfvec = ones(Float32,len)./len
    burnin = Int32(100)
    iterations = Int32(1000)
    xystd = Float32(.1)
    istd = Float32(10)
    split_std = Float32(1)
    bndpixels = Float32(0)
    
    # return [length(MATLAB.jvalue(arg)) for arg in args]

    mapn = matlab_DD_FlatBG(roi,psftype,σ_psf,
    θ_start,θ_step,len,pdfvec,
    burnin,iterations,
    xystd, istd,split_std,bndpixels)
  
    return [mapn.x,mapn.y,mapn.photons,mapn.σ_x,mapn.σ_y,mapn.σ_photons]
end

function matlab_DD_FlatBG_mex(args::Vector{MATLAB.MxArray})
    roi = MATLAB.jvalue(args[1])
    psftype = MATLAB.jvalue(args[2])
    σ_psf = MATLAB.jvalue(args[3])
    θ_start = MATLAB.jvalue(args[4])
    θ_step = MATLAB.jvalue(args[5])
    len = MATLAB.jvalue(args[6])
    pdfvec = vec(MATLAB.jvalue(args[7]))
    burnin = MATLAB.jvalue(args[8])
    iterations = MATLAB.jvalue(args[9])
    xystd = MATLAB.jvalue(args[10])
    istd = MATLAB.jvalue(args[11])
    split_std = MATLAB.jvalue(args[12])
    bndpixels = MATLAB.jvalue(args[13])
    
    # return [length(MATLAB.jvalue(arg)) for arg in args]

    e = 0
    mapn=0
    try
        mapn = matlab_DD_FlatBG(roi,psftype,σ_psf,
    θ_start,θ_step,len,pdfvec,
    burnin,iterations,
    xystd, istd,split_std,bndpixels)
    catch e
        return [@sprintf("%s",e)]
    end
    return [mapn.x,mapn.y,mapn.photons,mapn.σ_x,mapn.σ_y,mapn.σ_photons]
end

function matlab_DD_FlatBG(roi::Array{Float32,2},psftype::String,σ_psf::Float32,
    θ_start::Float32,θ_step::Float32,len::Int32,pdfvec::Vector{Float32},
    burnin::Int32,iterations::Int32,
    xystd::Float32, istd::Float32,split_std::Float32,bndpixels::Float32)

    sz = Int32(size(roi, 1))

    if psftype == "gauss"
        psf = PSF_gauss2D(σ_psf)
    end

    if psftype == "airy"
        psf = PSF_airy2D(σ_psf)
    end

    prior_photons = RJPrior(len, θ_start, θ_step, pdfvec)    
    data = ArrayDD_CPU(sz, roi)

    myRJ = RJStruct(sz, psf, xystd, istd, split_std, data, bndpixels, prior_photons)
    jumpprobability = [1,0,.1,.1,.1,.1] # Model with no bg 
    jumpprobability = jumpprobability / sum(jumpprobability)
    njumptypes = Int32(length(jumpprobability))

# create an RJMCMC structure with all model info
    acceptfuns = [accept_move,accept_bg,accept_add,accept_remove,accept_split,accept_merge] # array of functions
    propfuns = [propose_move,propose_bg,propose_add,propose_remove,propose_split,propose_merge] # array of functions
    myRJMCMC = ReversibleJumpMCMC.RJMCMCStruct(burnin, iterations, njumptypes, jumpprobability, propfuns, acceptfuns)

# create an intial state
    state1 = calcintialstate(myRJ)

## run chain. This is the call to the main algorithm
    mychain = ReversibleJumpMCMC.buildchain(myRJMCMC, myRJ, state1)
    Results_mapn = getmapn(mychain.states)
    return Results_mapn
end




