# Main Interface to BAMF 

"""
    function bamf_roi()

run BAMF on single ROI and return chain 

This is the main entry point to the BAMF algorithm

```
function bamf_roi(data::BAMFData, psf::MicroscopePSFs.PSF, prior_photons::Distributions.UnivariateDistribution, prior_background::Distributions.UnivariateDistribution;
    bndpixels::Real = 0.0,
    σ_xy::Real = 0.05,
    σ_split::Real = 0.5,
    σ_photons::Real = sqrt(mean(prior_photons)) / 10,
    σ_background::Real = sqrt(mean(prior_background)) / 10,
    burnin::Int=5000,
    iterations::Int=5000,
    jumpprobability = [.5, .1, .1, .1, .1, .1]
)
```

# Arguments
- `data::BAMFData`: sub region dataa
- `psf::MicroscopePSFs.PSF`: point spread function 
- `prior_photons::Distributions.UnivariateDistribution`: prior probability on photons/emitter/frame
- `prior_background::Distributions.UnivariateDistribution`: prior probability on bg photons/pixel

# Optional Keyword Arguments
- `bndpixels::Real = 0.0`: boundary region. +1 means allow positoins 1 pixel larger than ROI. 
- `σ_xy::Real = 0.05`: parameterizes MCMC jumps in x,y
- `σ_split::Real = 0.5`: parameterizes MCMC jumps in seperation in splitting
- `σ_photons::Real = sqrt(mean(prior_photons)) / 10`: parameterizes MCMC jumps in photons/emitter
- `σ_background::Real = sqrt(mean(prior_background)) / 10`: parameterizes MCMC jumps bg 
- `burnin::Int=5000`:
- `iterations::Int=5000`:
- `jumpprobability = [.5, .1, .1, .1, .1, .1]`: [move, bg, add,remove,split,merge]. Will be normalized internally 

`jumpprobability` can be used to fix number of emitters, bg, or emitter x,y,photons by setting jump probability to zero. 


"""
function bamf_roi(data::BAMFData, psf::MicroscopePSFs.PSF, prior_photons::Distributions.UnivariateDistribution, prior_background::Distributions.UnivariateDistribution;
    bndpixels::Real = 0.0,
    σ_xy::Real = 0.05,
    σ_split::Real = 0.5,
    σ_photons::Real = min(sqrt(mean(prior_photons)) / 10 , std(prior_photons) / 10),
    σ_background::Real = sqrt(mean(prior_background)) / 10,
    burnin::Int=5000,
    iterations::Int=5000,
    jumpprobability = [.5, .1, .1, .1, .1, .1]
)
    currentdata=deepcopy(data)
    println(typeof(currentdata))
    testdata=deepcopy(data)
    
    myRJ = RJStruct(data, psf, prior_photons, prior_background, bndpixels,
        σ_xy, σ_split, σ_photons, σ_background, currentdata, testdata)

    ## setup the RJMCMC.jl model
    # Jumptypes are: move, bg, add, remove, split, merge
    njumptypes = 6
    jumpprobability = jumpprobability / sum(jumpprobability)

    # create an RJMCMC structure with all model info
    acceptfuns = [accept_move,  accept_bg,  accept_add,  accept_remove,  accept_split,  accept_merge] #array of functions
    propfuns = [propose_move,  propose_bg,  propose_add,  propose_remove,  propose_split,  propose_merge] #array of functions
    myRJMCMC = ReversibleJumpMCMC.RJMCMCStruct(burnin, iterations, njumptypes, jumpprobability, propfuns, acceptfuns)

    #create an intial state
    state1 =  calcintialstate(myRJ)

    ## run chain. This is the call to the main algorithm
    return ReversibleJumpMCMC.buildchain(myRJMCMC, myRJ, state1)

end
