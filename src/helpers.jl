## Helper functions
"""
    randID(k::Int32)  

return random number between 1 and k
"""
function randID(k::Int32)
    return Int32(ceil(k * rand()))
end

"""
    likelihoodratio(m::Array{Float32}, mtest::Array{Float32}, d::Array{Float32})  

Generic Likelihood calculation using Poisson noise statistics
"""
function likelihoodratio(m::Array{Float32}, mtest::Array{Float32}, d::Array{Float32})
    nelem=prod(size(m))
    minmodel=1f-6 #to avoid division by zero and log of zero
    LLR = 0;
    for ii = 1:nelem
        LLR += m[ii] - mtest[ii] + d[ii] * log(max(mtest[ii],minmodel) / max(m[ii],minmodel));   
    end
    L = exp(LLR)
    if L < 0
        println(L, LLR)
    end
    return exp(LLR)
end

""""
    poissrnd!(d::Array{Float32})
In-place Poisson noise corruptor
"""
function poissrnd!(d::Array{Float32})
    nelem=prod(size(d))
    for nn = 1:nelem
        d[nn] = Float32(rand(Poisson(Float64(d[nn]))))
    end
end

# "An inexpensive Normal random number generator for CUDA"
function curandn()
    r=0;
    for ii=1:12
        r+=rand()
    end
    return r-6f0
end

"""
    calcintialstate(rjs::RJStruct, seed::Int32)  

calculates an intial state by finding the maximum of the data and places an emitter
at that location with an intensity selected from the prior. 

allows the input of an optional non-negative, integer random seed that generates from 
the Mersenne Twister library to produce deterministic initial states. If no integer is input, 
the function uses the OS provided entropy.

"""
function calcintialstate(rjs::RJStruct, seed::Int32=Int32(-1)) # find initial state for direct detection data 
    if seed == -1
    else
        Random.seed!(seed);
    end
    d = rjs.data.data
    state1 = StateFlatBg()
    state1.n = 1
    state1.bg = minimum(d)
    roimax = maximum(d)
    coords = findall(x -> x == roimax, d)
    state1.y[1] = coords[1].I[1]
    state1.x[1] = coords[1].I[2]
    state1.photons[1] = priorrnd(rjs.prior_photons)
    return state1
end

