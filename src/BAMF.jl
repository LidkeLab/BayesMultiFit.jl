module BAMF

using Plots
using ImageView
using Distributions 
using CUDA

# Jumptypes are:
# bg, move, add,remove, split, merge

## Data types

## StateFlatBg-------------
abstract type StateFlatBg end
mutable struct StateFlatBg_CUDA <: StateFlatBg # this gets saved in chain
    n::Int32
    x::CuArray{Float32}
    y::CuArray{Float32}
    photons::CuArray{Float32}
    bg::Float32
end
mutable struct StateFlatBg_CPU <: StateFlatBg # this gets saved in chain
    n::Int32
    x::Vector{Float32}
    y::Vector{Float32}
    photons::Vector{Float32}
    bg::Float32
end
StateFlatBg() = StateFlatBg_CPU(0, [0], [0], [0], 0)
StateFlatBg(n::Int32) = StateFlatBg_CPU(n, Vector{Float32}(undef,n),Vector{Float32}(undef,n), Vector{Float32}(undef,n), 0)
StateFlatBg(n::Int32, x::Vector{Float32},y::Vector{Float32}, photons::Vector{Float32},bg::Float32)=StateFlatBg_CPU(n, x,y,photons, bg)
StateFlatBg_CUDA(n::Int32) = StateFlatBg_CPU(n, CuArray{Float32}(undef,n), CuArray{Float32}(undef,n), CuArray{Float32}(undef,n), 0)

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
    return s
end
## ------------------


## ArrayDD----------
abstract type ArrayDD end
mutable struct ArrayDD_CUDA <: ArrayDD  # direct detection data 
    sz::Int32
    data::CuArray{Float32,2}
end
mutable struct ArrayDD_CPU <: ArrayDD  # direct detection data 
    sz::Int32
    data::Array{Float32,2}
end
ArrayDD_CUDA(sz) = ArrayDD(sz, CuArray{Float32}(undef, sz, sz))
ArrayDD(sz) = ArrayDD_CPU(sz, Array{Float32}(undef, sz, sz))
## ------------------

mutable struct RJStructDD # contains data and all static info for Direct Detecsionpassed to BAMF functions 
    sz::Int32
    σ::Float32
    xy_std::Float32
    I_std::Float32
    data::ArrayDD
end
RJStructDD(sz,σ,xy_std,I_std) = RJStructDD(sz, σ, xy_std, I_std, ArrayDD(sz))

## Helper functions
function randID(k::Int32)
    return Int32(ceil(k * rand()))
end

function poissrnd!(d::ArrayDD)
    for nn = 1:d.sz^2
        d.data[nn] = Float32(rand(Poisson(Float64(d.data[nn]))))
    end
end
function poissrnd(d::ArrayDD)
    out = ArrayDD(d.sz)
    for nn = 1:d.sz^2
        out.data[nn] = Float32(rand(Poisson(Float64(d.data[nn]))))
    end
    return out
end

function curandn()
    r=0;
    for ii=1:12
        r+=rand()
    end
    return r-6f0
end

function circleShape(h, k, r)
    θ = LinRange(0, 2 * π, 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end

## genmodel_2Dgauss_CUDA -------
function genmodel_2Dgauss!(s::StateFlatBg, sz::Int32, σ::Float32, model::Array{Float32,2})
    for ii = 1:sz
        for jj = 1:sz
            model[ii,jj] = s.bg + 1f-4
            for nn = 1:s.n
                model[ii,jj] += s.photons[nn] / (2 * π * σ^2) *
                exp(-(ii - s.y[nn])^2 / (2 * σ^2)) *
                exp(-(jj - s.x[nn])^2 / (2 * σ^2))
            end
        end
    end
end

function genmodel_2Dgauss_CUDA!(s_n::Int32,s_x, s_y, 
    s_photons,s_bg::Float32, sz::Int32, σ::Float32, model)
     #note that the 2d array is linearized and using 1-based indexing in kernel
     ii = blockIdx().x
     jj = threadIdx().x 
     
     idx=(ii-1)*sz+jj
     model[idx] = s_bg + 1f-4
     for nn = 1:s_n
         model[idx] += s_photons[nn] / (2 * π * σ^2) *
                 exp(-(ii - s_y[nn])^2 / (2 * σ^2)) *
                 exp(-(jj - s_x[nn])^2 / (2 * σ^2))
     end
     return nothing
end

function genmodel_2Dgauss_CUDA!(s::StateFlatBg, sz::Int32, σ::Float32, model)
    #note that the 2d array is linearized and using 0-based indexing in kernel
    ii = blockIdx().x
    jj = threadIdx().x 
    
    idx=(ii-1)*sz+jj
    model[idx] = s.bg + 1f-4
    for nn = 1:s.n
        model[idx] += s.photons[nn] / (2 * π * σ^2) *
                exp(-(ii - s.y[nn])^2 / (2 * σ^2)) *
                exp(-(jj - s.x[nn])^2 / (2 * σ^2))
    end
    return nothing
end

genmodel_2Dgauss!(m::StateFlatBg,RJStructDD,model::ArrayDD) =
    genmodel_2Dgauss!(m, RJStructDD.sz, RJStructDD.σ, model.data)
function genmodel_2Dgauss!(m::StateFlatBg, sz::Int32, σ::Float32, model::ArrayDD)
    genmodel_2Dgauss!(m, sz, σ, model.data)
end
function genmodel_2Dgauss!(s::StateFlatBg, sz::Int32, σ::Float32, model::CuArray{Float32,2}) 
    @cuda threads=sz blocks=sz genmodel_2Dgauss_CUDA!(s.n,s.x,s.y,s.photons,s.bg, sz, σ, model)
end

# ---------------------

function calcintialstate(rjs::RJStructDD) # find initial state for direct detection data 
    d = rjs.data.data
    state1 = StateFlatBg()
    state1.n = 1
    state1.bg = minimum(d)
    roimax = maximum(d)
    coords = findall(x -> x == roimax, d)
    state1.y[1] = coords[1].I[1]
    state1.x[1] = coords[1].I[2]
    state1.photons[1] = (roimax - state1.bg) * 2 * π * rjs.σ^2
    return state1
end


## likelihoodratio --------------
function likelihoodratio(m::ArrayDD, mtest::ArrayDD, d::Array{Float32,2})
    LLR = 0;
    for ii = 1:m.sz * m.sz
        LLR += m.data[ii] - mtest.data[ii] + d.data[ii] * log(mtest.data[ii] / m.data[ii]);   
    end
    L = exp(LLR)
    if L < 0
        println(L, LLR)
    end
    return exp(LLR)
end

function likelihoodratio(sz::Int32, m::Array{Float32,2}, mtest::Array{Float32,2}, d::Array{Float32,2})
    LLR = 0;
    for ii = 1:sz * sz
        LLR += m[ii] - mtest[ii] + d[ii] * log(mtest[ii] / m[ii]);   
    end
    L = exp(LLR)
    if L < 0
        println(L, LLR)
    end
    return exp(LLR)
end

function likelihoodratio(m::ArrayDD, mtest::ArrayDD, d::ArrayDD)
     return likelihoodratio(m.sz,m.data,mtest.data,d.data)
end

function likelihoodratio(sz,m::CuArray{Float32,2}, mtest::CuArray{Float32,2}, d::CuArray{Float32,2})
    LLR=CUDA.zeros(1)
    # LLR[1]=0;
    @cuda threads=sz blocks=sz likelihoodratio_CUDA!(sz,m,mtest,d,LLR)
    L = exp(LLR[1])
    return L
end

function likelihoodratio_CUDA!(sz::Int32,m, mtest, d,LLR)
    ii = blockIdx().x
    jj = threadIdx().x 
    idx=(ii-1)*sz+jj
    llr=m[idx] - mtest[idx] + d[idx] * log(mtest[idx] / m[idx])
    CUDA.atomic_add!(pointer(LLR,1), llr)
    return nothing
end

## ---------------------


## Acceptance probability calculations

function accept_move(rjs::RJStructDD, currentstate::StateFlatBg, teststate::StateFlatBg)   
    roi = ArrayDD(rjs.sz)
    roitest = ArrayDD(rjs.sz)
    genmodel_2Dgauss!(currentstate, rjs, roi)
    genmodel_2Dgauss!(teststate, rjs, roitest)
    α = likelihoodratio(roi, roitest, rjs.data)
    return α
end

function accept_bg(rjs::RJStructDD, currentstate::StateFlatBg, teststate::StateFlatBg)

    return rand()
end

function accept_add()
    return rand()
end

function accept_remove()
    return rand()
end

function accept_split()
    return rand()
end

function accept_merge()
    return rand()
end

## State proposal Functions

function propose_move(rjs::RJStructDD, currentstate::StateFlatBg)
    teststate = StateFlatBg(currentstate)
    # get an emitter
    ID = randID(currentstate.n)
    # move the emitter
    move_emitter!(ID,teststate.x,teststate.y,teststate.photons,rjs.xy_std,rjs.I_std)
    return teststate
end
function move_emitter!(ID::Int32,x::CuArray,y::CuArray,photons::CuArray,xy_std::Float32,i_std::Float32) 
    @cuda move_emitter_CUDA!(ID,x,y,photons,xy_std,i_std)
end
function move_emitter_CUDA!(ID::Int32,x,y,photons,xy_std::Float32,i_std::Float32)
    x[ID]+=xy_std*curandn()
    y[ID]+=xy_std*curandn()
    photons[ID]+=i_std*curandn()
    return nothing
end
function move_emitter!(ID::Int32,x::Vector{Float32},y::Vector{Float32},photons::Vector{Float32},xy_std::Float32,i_std::Float32) 
    x[ID]+=xy_std*randn()
    y[ID]+=xy_std*randn()
    photons[ID]+=i_std*randn()
    return nothing
end



function propose_bg()
    return rand()
end

function propose_add()
    return rand()
end

function propose_remove()
    return rand()
end

function propose_split()
    return rand()
end

function propose_merge()
    return rand()
end


## Analysis and Display

function histogram2D(states::Vector{Any}, sz::Int32, zoom::Int32)
    # count number of emitters 

    nemitters = 0;
    for nn = 1:length(states) 
        nemitters += states[nn].n
    end
    
    x = Vector{Float32}(undef, nemitters)
    y = Vector{Float32}(undef, nemitters)
 
    cnt = 0;
    for ss = 1:length(states) 
        for nn = 1:states[ss].n
            cnt += 1  
            x[cnt] = states[ss].x[nn]
            y[cnt] = states[ss].y[nn]
        end
    end   

    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    histplot = histogram2d(x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=true) 
    return histplot
end

function histogram2D(states::Vector{Any}, sz::Int32, zoom::Int32, d::ArrayDD, truestate::StateFlatBg)
     # count number of emitters 
     
      # count number of emitters 
    nemitters = 0;
    for nn = 1:length(states) 
        nemitters += states[nn].n
    end
    
    x = Vector{Float32}(undef, nemitters)
    y = Vector{Float32}(undef, nemitters)
 
    cnt = 0;
    for ss = 1:length(states) 
        for nn = 1:states[ss].n
            cnt += 1  
            x[cnt] = states[ss].x[nn]
            y[cnt] = states[ss].y[nn]
        end
    end   

    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    fig = heatmap(d.data, color=:greys)
    histogram2d!(fig, x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false) 
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue)  
    end
    return fig
end


end



