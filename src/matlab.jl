## This provides a mex-style interface for calling from MATLAB via Mex.jl

using MATLAB

function matlab_DD_FlatBG_mex(args::Vector{MATLAB.MxArray})
    
    roi=MATLAB.jvalue(args[1])
    # psftype=MATLAB.jvalue(2)
    # σ_psf=MATLAB.jvalue(3)
    # θ_start=MATLAB.jvalue(4)
    # θ_step=MATLAB.jvalue(5)
    # len=MATLAB.jvalue(6)
    # pdfvec=MATLAB.jvalue(7)
    # burnin=MATLAB.jvalue(8)
    # iterations=MATLAB.jvalue(9)
    # xystd=MATLAB.jvalue(10)
    # istd=MATLAB.jvalue(11)
    # split_std=MATLAB.jvalue(12)
    # bndpixels=MATLAB.jvalue(13)
    
    # mapn=matlab_DD_FlatBG(roi,psftype,σ_psf,
    # θ_start,θ_step,len,pdfvec,
    # burnin,iterations,
    # xystd, istd,split_std,bndpixels)

    return [roi]
end

function matlab_DD_FlatBG(roi::Array{Float32},psftype::Vector{Char},σ_psf::Float32,
    θ_start::Float32,θ_step::Float32,len::Int32,pdfvec::Vector{Float32},
    burnin::Int32,iterations::Int32,
    xystd::Float32, istd::Float32,split_std::Float32,bndpixels::Float32)

    sz=Int32(size(roi,1))

    if psftype==collect("gauss")
        psf=PSF_gauss2D(σ_psf)
    end

    if psftype==collect("airy")
        psf=PSF_airy2D(σ_psf)
    end

    prior_photons=RJPrior(len,θ_start,θ_step,pdfvec)    
    data=ArrayDD_CPU(sz,roi)

    myRJ = RJStruct(sz, psf, xystd, istd, split_std, data, bndpixels, prior_photons)
    jumpprobability = [1,0,.1,.1,.1,.1] # Model with no bg 
    jumpprobability = jumpprobability / sum(jumpprobability)
    njumptypes=Int32(length(jumpprobability))

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




