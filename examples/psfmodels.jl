## Show the calculation of Gaussian and Airy PSF models

#include("../src/BayesMultiFit.jl")
#import BAMF: RJMCMC
using BayesMultiFit
BAMF=BayesMultiFit
using ImageView
using Plots
ImageView.closeall()


## Setup
sz=Int32(32)

## setup point source
n=Int32(1)
x=[Float32(sz/2f0)]
y=[Float32(sz/2f0)]
photons=[1f0]
bg=0f0
pointsource=BAMF.StateFlatBg(n,x,y,photons,bg)

## 2D Gaussian PSF
σ=Float32(1.3)
psf=BAMF.PSF_gauss2D(σ)
gausspsf=BAMF.ArrayDD(sz)
BAMF.genmodel!(pointsource,psf,gausspsf)
imshow(gausspsf.data)
println(("integrated value: " ,sum(gausspsf.data)))

## 2D Airy PSF 
pixelsize=.05
nₐ=1.4
λ=.6
ν=Float32(2π*nₐ/λ)*pixelsize
psf=BAMF.PSF_airy2D(ν)
airypsf=BAMF.ArrayDD(sz)
BAMF.genmodel!(pointsource,psf,airypsf)
imshow(airypsf.data)
println(("integrated value: " ,sum(airypsf.data)))

