module BAMF

using Plots
using ImageView
using Distributions 
using CUDA

## Data types
#using Revise

include("bamftypes.jl")
include("helpers.jl")
include("gauss2D.jl")
include("propose.jl")
include("accept.jl")
include("display.jl")



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
    state1.photons[1] = (roimax - state1.bg) * max2int(rjs.psf) 
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





end



