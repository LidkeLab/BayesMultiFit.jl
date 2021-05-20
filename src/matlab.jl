## This provides a mex-style interface for calling from MATLAB via Mex.jl




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




