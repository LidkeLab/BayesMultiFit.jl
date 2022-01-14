## Generic types and methods



"""
    PSF

Abstract type.  Specific psf types will inherit from PSF
"""
abstract type PSF end


"""
    BAMFState

Abstract type.  Specific state types will inherit from BAMFState.  

States hold the current state of the RJMCMC Chain (i.e. θ) and can also be used to
define a true underlying state that generated the data. 
"""
abstract type BAMFState end


"""
    BAMFData

BAMFData is an abstract type.  Specific data types will inherit from BAMFData.  

The most common example is 'directdetection'  
"""
abstract type BAMFData end



"""
    StateFlatBg<: BAMFState

Abstract type with children that implement CPU and GPU variants 
"""    
abstract type StateFlatBg <: BAMFState end


mutable struct StateFlatBg_CUDA <: StateFlatBg # this gets saved in chain
    n::Int
    x::CuArray{Float32}
    y::CuArray{Float32}
    photons::CuArray{Float32}
    bg::Float32
end

"""
    StateFlatBg_CPU(n::Int, x::Vector{Float32}, y::Vector{Float32}, photons::Vector{Float32}, bg::Float32)

State of a model that has x,y positions, total integrated intensity and a flat background offset. 
"""
mutable struct StateFlatBg_CPU <: StateFlatBg # this gets saved in chain
    n::Int
    x::Vector{Float32}
    y::Vector{Float32}
    photons::Vector{Float32}
    bg::Float32
end

mutable struct StateFlatBg_Results <: StateFlatBg # this gets saved in chain
    n::Int
    x::Vector{Float32}
    y::Vector{Float32}
    photons::Vector{Float32}
    σ_x::Vector{Float32}
    σ_y::Vector{Float32}
    σ_photons::Vector{Float32}
    bg::Float32
end


StateFlatBg() = StateFlatBg_CPU(0, [0], [0], [0], 0)
StateFlatBg(n::Int) = StateFlatBg_CPU(n, Vector{Float32}(undef,n),Vector{Float32}(undef,n), Vector{Float32}(undef,n), 0)
StateFlatBg(n::Int, x::Vector{Float32},y::Vector{Float32}, photons::Vector{Float32},bg::Float32)=StateFlatBg_CPU(n, x,y,photons, bg)
StateFlatBg_CUDA(n::Int) = StateFlatBg_CPU(n, CuArray{Float32}(undef,n), CuArray{Float32}(undef,n), CuArray{Float32}(undef,n), 0)
StateFlatBg_Results(n::Int) = StateFlatBg_Results(n, Vector{Float32}(undef,n),
    Vector{Float32}(undef,n), Vector{Float32}(undef,n),Vector{Float32}(undef,n),Vector{Float32}(undef,n), Vector{Float32}(undef,n), 0)
StateFlatBg_Results(state::StateFlatBg_CPU) = StateFlatBg_Results(state.n,state.x,state.y,state.photons,
                        fill!(similar(state.x),NaN),fill!(similar(state.y),NaN),fill!(similar(state.photons),NaN),state.bg)

function StateFlatBg_CUDACopy!(myempty,myfull)
    idx=threadIdx().x
    myempty[idx]=myfull[idx]
    return nothing
end
function StateFlatBg(sf::StateFlatBg_CUDA) # make a deep copy
    s = StateFlatBg_CUDA(sf.n)
    @cuda threads=s.n StateFlatBg_CUDACopy!(s.x,sf.x)
    @cuda threads=s.n StateFlatBg_CUDACopy!(s.y,sf.y)
    @cuda threads=s.n StateFlatBg_CUDACopy!(s.photons,sf.photons)
    return s
end
function StateFlatBg(sf::StateFlatBg_CPU) # make a deep copy
    s = StateFlatBg(sf.n)
    for nn=1:s.n
        s.x[nn]=sf.x[nn]
        s.y[nn]=sf.y[nn]
        s.photons[nn]=sf.photons[nn]
    end
    s.bg=sf.bg
    return s
end

function addemitter!(s::StateFlatBg,x::Float32,y::Float32,photons::Float32)
    s.n=s.n+1
    push!(s.x,x)
    push!(s.y,y)
    push!(s.photons,photons)
end

function removeemitter!(s::StateFlatBg,idx::Int)
    s.n=s.n-1
    deleteat!(s.x,idx)
    deleteat!(s.y,idx)
    deleteat!(s.photons,idx)
end

function findclosest(s::StateFlatBg,idx::Int)
    if (s.n<2)||(idx<1) return 0 end
    
    mindis=1f5 #big
    nn=1
    for nn=1:s.n
        if nn!=idx
            dis=(s.x[idx]-s.x[nn])^2+(s.y[idx]-s.y[nn])
            if dis<mindis
                mindis=dis
                idx=nn
            end
        end
    end
    return nn
end

function findother(s::StateFlatBg,idx::Int)
    if (s.n<2)||(idx<1) return 0 end
    
    nn=Int(ceil(s.n*rand()))
    while nn==idx
        nn=Int(ceil(s.n*rand()))
    end

    return nn
end



## ------------------

## ------------------


## RJPrior -----------------
mutable struct RJPrior
    sz::Int
    θ_start::Float32
    θ_step::Float32
    pdf::Vector{Float32}
end
RJPrior()=RJPrior(1,1,1,[1])

function priorrnd(rjp::RJPrior)
    r=rand()
    mysum=rjp.pdf[1] 
    nn=1
    while (mysum<r)&&(nn<rjp.sz)
        nn+=1
        mysum+=rjp.pdf[nn]
    end
    return Float32((nn-1)*rjp.θ_step+rjp.θ_start)
end

""""
    priorpdf(rjp::RJPrior,θ)   

calculate PDF(θ)
"""
function priorpdf(rjp::RJPrior,θ)
nn=round((θ-rjp.θ_start)/rjp.θ_step)
nn=Int(max(1,min(nn,rjp.sz)))
return rjp.pdf[nn]
end


"""
    RJStruct

Holds the data to be analyzed, the priors, the PSF model, and parameters used in the RJMCMC steps. 
"""
mutable struct RJStruct # contains data and all static info for Direct Detection passed to BAMF functions 
    data::BAMFData
    psf::MicroscopePSFs.PSF
    prior_photons::Distributions.UnivariateDistribution
    prior_background::Distributions.UnivariateDistribution   
    bndpixels::Real
    σ_xy::Real
    σ_split::Real
    σ_photons::Real         
    σ_background::Real
    modeldata::BAMFData
    testdata::BAMFData
end


# RJStruct(sz,psf,xy_std,I_std,split_std) = 
#     RJStruct(sz, psf, xy_std, I_std, split_std,ArrayDD(sz),2,RJPrior(),ArrayDD(sz),ArrayDD(sz))
# RJStruct(sz,psf,xy_std,I_std,split_std,data::BAMFData) = 
#     RJStruct(sz, psf, xy_std, I_std, split_std,data,2,RJPrior(),deepcopy(data),deepcopy(data))
# RJStruct(sz,psf,xy_std,I_std,split_std,data::BAMFData,bndpixels,prior_photons) = 
#     RJStruct(sz, psf, xy_std, I_std, split_std,data,bndpixels,prior_photons,deepcopy(data),deepcopy(data))



"""
    genBAMFData(rjs::RJStruct)

generate an empty BAMFData stucture for data type in RJStruct
"""
function genBAMFData(rjs::RJStruct)
    return genBAMFData(rjs.data)
end











