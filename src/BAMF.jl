module BAMF

using Plots
using ImageView
using Distributions 
using CUDA



# Jumptypes are:
# bg, move, add,remove, split, merge

## Data types
println(pwd())
#using Revise
include("bamftypes.jl")

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

function calcresiduum(model::ArrayDD,data::ArrayDD)
    residuum = ArrayDD(model.sz);
    for nn=1:model.sz*model.sz
        residuum.data[nn]=data.data[nn]-model.data[nn]
    end
    return residuum
end


function makepdf!(a::ArrayDD)
#make all elements sum to 1
mysum=0;
for nn=1:a.sz*a.sz
    a.data[nn]=max(0,a.data[nn])
    mysum+=a.data[nn]
end
for nn=1:a.sz*a.sz
    a.data[nn]/=mysum
end
end

function makecdf!(a::ArrayDD)
#make the array a normalized CDF
makepdf!(a)
for nn=2:a.sz*a.sz
    a.data[nn]+=a.data[nn-1];
end
end

function arrayrand!(a::ArrayDD)
#pull random number from pdf array
#this converts input to cdf
    makecdf!(a)
    r=rand()
    nn=1
    while a.data[nn]<r
        nn+=1
    end
    ii=rem(nn,a.sz)
    jj=ceil(nn/a.sz)
    return ii,jj 
end
    
function arraypdf(a::ArrayDD,ii::Int32,jj::Int32)
#calculate probability at pixel index
mysum=0;
for nn=1:a.sz*a.sz
    mysum+=max(1,a.data[nn]);
end
  return max(1,a.data[ii,jj])/mysum
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
    if minimum(teststate.photons)<0
        println(currentstate)
        println(teststate)
    end

    LR=likelihoodratio(roi, roitest, rjs.data)
    PR=1
    for nn=1:teststate.n
        PR*=priorpdf(rjs.prior_photons,teststate.photons[nn])/
            priorpdf(rjs.prior_photons,currentstate.photons[nn])        
    end
    α = PR*LR
    return α
end

function accept_bg(rjs::RJStructDD, currentstate::StateFlatBg, teststate::StateFlatBg)

    return rand()
end

function accept_add(rjs::RJStructDD, currentstate::StateFlatBg, teststate::StateFlatBg)
    roi = ArrayDD(rjs.sz)
    roitest = ArrayDD(rjs.sz)
    genmodel_2Dgauss!(currentstate, rjs, roi)
    genmodel_2Dgauss!(teststate, rjs, roitest)
    LLR = likelihoodratio(roi, roitest, rjs.data)
    #proposal probability
    residuum=calcresiduum(roitest,rjs.data)   
    jj=Int32(min(roi.sz,max(1,round(teststate.x[end]))))
    ii=Int32(min(roi.sz,max(1,round(teststate.y[end]))))
    p=arraypdf(roitest,ii,jj)
    α = LLR*(roi.sz+rjs.bndpixels)^2/p
    #println(("add: ",α,ii,jj,teststate.photons[end]))
    return α
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
    move_emitter!(ID,teststate.x,teststate.y,teststate.photons,rjs)
    return teststate
end
function move_emitter!(ID::Int32,x::CuArray,y::CuArray,photons::CuArray,rjs::RJStructDD) 
    @cuda move_emitter_CUDA!(ID,x,y,photons,rjs.xy_std,rjs.I_std)
end
function move_emitter_CUDA!(ID::Int32,x,y,photons,xy_std::Float32,i_std::Float32)
    x[ID]+=xy_std*curandn()
    y[ID]+=xy_std*curandn()
    photons[ID]+=i_std*curandn()
    photons[ID]=max(0,photons[ID])
    return nothing
end
function move_emitter!(ID::Int32,x::Vector{Float32},y::Vector{Float32},photons::Vector{Float32},rjs::RJStructDD) 
    x[ID]+=rjs.xy_std*randn()
    y[ID]+=rjs.xy_std*randn()
    photons[ID]+=rjs.I_std*randn()
    photons[ID]=max(rjs.prior_photons.θ_start,photons[ID])
    return nothing
end



function propose_bg()
    return rand()
end


## Add and remove -------------

function propose_add(rjs::RJStructDD, currentstate::StateFlatBg)

    teststate = StateFlatBg(currentstate)
    
    #calc residum
    roi = ArrayDD(rjs.sz)
    roitest = ArrayDD(rjs.sz)
    genmodel_2Dgauss!(currentstate, rjs, roi)
    genmodel_2Dgauss!(teststate, rjs, roitest)
    residuum=calcresiduum(roitest,rjs.data)   
    #pick pixel
    ii,jj=arrayrand!(residuum)

    #pick intensity from prior
    photons=priorrnd(rjs.prior_photons)
    if photons<0
        println(photons)
    end
    #add new emitter
    addemitter!(teststate,Float32(ii),Float32(jj),photons)

    return teststate
end


function propose_remove()
    return rand()
end

## ---------------------------




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



