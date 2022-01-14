## This shows the simplest use of BAMF with a Gaussian PSF

using Revise
using BayesMultiFit
BAMF=BayesMultiFit
using ReversibleJumpMCMC
const RJMCMC = ReversibleJumpMCMC
using Plots
using Distributions
using MicroscopePSFs
const PSF=MicroscopePSFs

plot_flag = true

## simulation config
n=6             # number of emitters
μ=1000          # mean photons per emitter               
sz=16           # ROI size in pixels
iterations=5000 # RJMCMC iterations
burnin=5000     # RJMCMC iterations for burn-in

## PSF config
σ=1.3 # Gaussian PSF Sigma in Pixels
pixelsize=1.0
psf=PSF.Gauss2D(σ,pixelsize)

## setup prior distribution on intensity
α=4.0
θ=μ/α
prior_photons=Gamma(α,θ)
pdf_x=range(1,quantile(prior_photons,.999),length=1024)
mypdf=pdf(prior_photons,pdf_x)
if plot_flag
    plt=plot(pdf_x,mypdf)
    display(plt)
end

## setup prior on background
prior_background=Gamma(1.0,1.0)


# set emitter positions and intensity
x=sz*rand(Float32,n)
y=sz*rand(Float32,n)

photons=Float32.(rand(prior_photons,n))
bg=Float32(rand(prior_background))
bg=1.0f0
datastate=BAMF.StateFlatBg(n,x,y,photons,bg)

## Create synthetic data
data=BAMF.ArrayDD(sz)      
BAMF.genmodel!(datastate,psf,data)
BAMF.poissrnd!(data.data)

if plot_flag
    heatmap(data.data)  #look at data  
end

## run chain. This is the call to the main algorithm
mychain=BAMF.bamf_roi(data,psf,prior_photons,prior_background)


## Display
if plot_flag
    plotly()
    zm=4
    plt=BAMF.histogram2D(mychain.states,sz,zm,datastate)
    display(plt)
end

map_n,posterior_n,traj_n=BAMF.getn(mychain.states)
if plot_flag
    plt2=plot(traj_n)
    display(plt2)
    out=BAMF.showoverlay(mychain.states,data,psf)
    plt=heatmap(out[:,:,end]')
    display(plt)
end

## MAPN Results
states_mapn,n=BAMF.getmapnstates(mychain.states)
if plot_flag
    plt=BAMF.histogram2D(states_mapn,sz,zm,datastate)
    display(plt)
end

Results_mapn=BAMF.getmapn(mychain.states)
if plot_flag
    gr()
    plt=BAMF.plotstate(datastate,Results_mapn;fig=plot())
    display(plt)
end




