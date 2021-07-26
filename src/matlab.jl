## This provides a mex-style interface for calling from MATLAB via Mex.jl

using MATLAB
using Printf

global matlab_chain

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
    pdfvec = ones(Float32, len) ./ len
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
    mapn = 0
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
    global matlab_chain
    matlab_chain = ReversibleJumpMCMC.buildchain(myRJMCMC, myRJ, state1)
    Results_mapn = getmapn(matlab_chain.states)
    return Results_mapn
end

function matlab_SLIVER_FlatBG_mex(args::Vector{MATLAB.MxArray})
    
    n = length(MATLAB.jvalue(args[3]))
    if n == 1 #force back to vectors
        roi = reshape(MATLAB.jvalue(args[1]),Val(3))
        meastype = [MATLAB.jvalue(args[2])]
        invx = [MATLAB.jvalue(args[3])]
        invy = [MATLAB.jvalue(args[4])]
    else
        roi = MATLAB.jvalue(args[1])
        meastype = MATLAB.jvalue(args[2])
        invx = MATLAB.jvalue(args[3])
        invy = MATLAB.jvalue(args[4])
    end

    psftype = MATLAB.jvalue(args[5])
    σ_psf = MATLAB.jvalue(args[6])
    θ_start = MATLAB.jvalue(args[7])
    θ_step = MATLAB.jvalue(args[8])
    len = MATLAB.jvalue(args[9])
    pdfvec = vec(MATLAB.jvalue(args[10]))
    burnin = MATLAB.jvalue(args[11])
    iterations = MATLAB.jvalue(args[12])
    xystd = MATLAB.jvalue(args[13])
    istd = MATLAB.jvalue(args[14])
    split_std = MATLAB.jvalue(args[15])
    bndpixels = MATLAB.jvalue(args[16])
    
    # return [length(MATLAB.jvalue(arg)) for arg in args]

    e = 0
    mapn = 0
    #try
        mapn = matlab_SLIVER_FlatBG(roi,meastype,invx,invy,psftype,σ_psf,
    θ_start,θ_step,len,pdfvec,
    burnin,iterations,
    xystd, istd,split_std,bndpixels)
    #catch e
    #    return [@sprintf("%s",e)]
    #end
    return [mapn.x,mapn.y,mapn.photons,mapn.σ_x,mapn.σ_y,mapn.σ_photons]
end


function matlab_SLIVER_FlatBG(roi::Array{Float32,3},meastype::Vector{Int32},invx::Vector{Float32},invy::Vector{Float32},
    psftype::String,σ_psf::Float32,
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
    data = DataSLIVER(sz, meastype)
    data.data = roi
    data.invx = invx
    data.invy = invy


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
    global matlab_chain
    matlab_chain = ReversibleJumpMCMC.buildchain(myRJMCMC, myRJ, state1)
    Results_mapn = getmapn(matlab_chain.states)
    return Results_mapn
end

"""
    pickpsf(psf::String, σ_psf::Float32)

constructs a PSF object of the type indicated by the string psf with the internal value σ_psf.
"""
function pickpsf(psf::String, σ_psf::Float32)
    if psf == "airy"
        psft = PSF_airy2D
    elseif psf == "gauss"
        psft = PSF_gauss2D
    end
    return pickpsf(psft, σ_psf)
end

function pickpsf(psft::Type{T}, σ_psf::Float32) where {T <: PSF}
    return psft(σ_psf)
end

"""
    pickMeasType(meastype::Int32, invx::Float32, invy::Float32)

returns a tuple of the MeasType appropriate to the integer and a tuple containing the information required to make that measurement
"""
function pickMeasType(meastype::Int32, invx::Float32, invy::Float32)
    return pickMeasType(Val(meastype), invx, invy)
end

function pickMeasType(::Val{Int32(1)}, invx::Float32, invy::Float32)
    return (DDMeasType, ())
end

function pickMeasType(::Val{Int32(2)}, invx::Float32, invy::Float32)
    return (SLIVERMeasType, (invx, invy))
end

function pickMeasType(meastype::Int32, invx::Float32, invy::Float32, inttime::Float32)
    e1, e2 = pickMeasType(meastype, invx, invy)
    return (e1, e2, inttime)
end

"""
    genMeasTypelist(meastypes::Vector{Int32}, invx::Vector{Float32}, invy::Vector{Float32}, inttime::Vector{Float32})

returns a vector of appropriate MeasType structures with the appropriate integration times and the inversion point info.
"""
function genMeasTypelist(meastypes::Vector{Int32}, invx::Vector{Float32}, invy::Vector{Float32}, inttime::Vector{Float32})
    n = length(meastypes)
    if n == length(invx) && n == length(invy)
        return genMeasTypelist([pickMeasType(meastypes[ii], invx[ii], invy[ii], inttime[ii]) for ii in 1:n])
    end
end

function genMeasTypelist(meastypes::Vector{Int32}, invx::Vector{Float32}, invy::Vector{Float32})
    n = length(meastypes)
    inttime = Float32(1/n)
    if n == length(invx) && n == length(invy)
        return genMeasTypelist([pickMeasType(meastypes[ii], invx[ii], invy[ii], inttime) for ii in 1:n])
    end
end
    

"""
    matlab_Adapt_FlatBG_mex(args::Vector{MxArray})

passes the matlab input through julia to return a vector containing the predicted model information.
"""
function matlab_Adapt_FlatBG_mex(args::Vector{MATLAB.MxArray})

    names = ["roi", "meastype", "invx", "invy", "psftype", "σ_psf", "θ_start", "θ_step", "len", "pdfvec", "burnin", "iterations", "xystd", "istd", "split_std", "bndpixels"]
    argdict = Dict(names[ii] => MATLAB.jvalue(args[ii]) for ii in 1:length(names))

    return matlab_Adapt_FlatBG(argdict)
end

function matlab_Adapt_FlatBG(argdict::Dict{String, T}) where {T<:Any}
    # reshape pdfvec to appropriate dimensions
    get!(argdict, "pdfvec", vec(pop!(argdict, "pdfvec")))

    # generate appropriate ArrayData structure
    vars = ["meastype", "invx", "invy"]
    meastype, invx, invy=ntuple(i->get(argdict, vars[i], "no entry"), 3)
    meastypelist = vec(genMeasTypelist(meastype, invx, invy))
    sz, = size(get(argdict, "roi", "no roi"), 1)
    data = AdaptData(Int32(sz), meastypelist)
    data.data = reshape(get(argdict, "roi", "no roi"), size(data.data))
    
    # generate appropriate PSF and RJStruct
    psf= pickpsf(get(argdict, "psftype", "no psftype"), get(argdict, "σ_psf", "no σ_psf"))
    vars= ["xystd", "istd", "split_std", "bndpixels", "len", "θ_start", "θ_step", "pdfvec"]
    xystd, istd, split_std, bndpixels, len, θ_start, θ_step, pdfvec= ntuple(i->get(argdict, vars[i], "no entry"), 8)
    prior_photons = RJPrior(len, θ_start, θ_step, pdfvec)
    myRJ = RJStruct(sz, psf, xystd, istd, split_std, data, bndpixels, prior_photons)
    jumpprobability = [1f0,0f0,1f-1,1f-1,1f-1,1f-1] # Model with no bg 
    jumpprobability = jumpprobability / sum(jumpprobability)
    njumptypes = Int32(length(jumpprobability))

    
    # create an RJMCMC structure with all model info
    acceptfuns = [accept_move,accept_bg,accept_add,accept_remove,accept_split,accept_merge] # array of functions
    propfuns = [propose_move,propose_bg,propose_add,propose_remove,propose_split,propose_merge] # array of functions
    myRJMCMC = ReversibleJumpMCMC.RJMCMCStruct(get(argdict, "burnin", "no burnin"), get(argdict, "iterations", "no iterations"), njumptypes, jumpprobability, propfuns, acceptfuns)
    
    # create an initial state
    state1 = calcintialstate(myRJ)

## run chain. This is the call to the main algorithm
    global matlab_chain
    matlab_chain = ReversibleJumpMCMC.buildchain(myRJMCMC, myRJ, state1)
    mapn = getmapn(matlab_chain.states)
    return [mapn.x,mapn.y,mapn.photons,mapn.σ_x,mapn.σ_y,mapn.σ_photons]
end



