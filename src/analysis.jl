# Analysis

using Clustering
using Statistics
using StatsBase

"""
x,y,photons=getxy(states::Vector{Any})

returns a vector of all x,y,photons over all states
"""
function getxy(states::Vector{Any})
    # count number of emitters 
    nemitters = 0;
    for nn = 1:length(states) 
        nemitters += states[nn].n
    end
    
    x = Vector{Float32}(undef, nemitters)
    y = Vector{Float32}(undef, nemitters)
    photons = Vector{Float32}(undef, nemitters)

    cnt = 0;
    for ss = 1:length(states) 
        for nn = 1:states[ss].n
            cnt += 1  
            x[cnt] = states[ss].x[nn]
            y[cnt] = states[ss].y[nn]
            photons[cnt] = states[ss].photons[nn]
        end
    end   
    return x,y,photons 
end



function getn(states::Vector{Any})
    maxemitters = 0
    nstates=length(states) 

    traj_n=Vector{Int32}((undef),nstates)
    for nn = 1:nstates
        traj_n[nn]=states[nn].n
        maxemitters=max(maxemitters,states[nn].n)
    end

    posterior_n=zeros(Float32, maxemitters+1) 
    for nn = 1:nstates
        posterior_n[states[nn].n+1]+=states[nn].n
    end
    posterior_n=posterior_n/sum(posterior_n)

    cntmax = maximum(posterior_n)
    coords = findall(x -> x == cntmax, posterior_n)
    map_n=Int(coords[1]-1)

    return map_n,posterior_n,traj_n
end


"""
states_mapn=getmapn(states::Vector{Any})

returns a vector of states that where states.n == map_n
"""
function getmapnstates(states::Vector{Any})
    map_n,posterior_n,traj_n=getn(states)

    nstates=length(states)
    idx=zeros(Bool,nstates)

    for ii=1:length(states)
        if states[ii].n==map_n
            idx[ii]=true
        end    
    end
    
    return states[idx],map_n
end

"""
    clusterstates(states::Vector{Any},n::Int32)

uses kmeans clustering group x,y locations into n clusters
and returns a State_Results structure. 

"""
function clusterstates(states::Vector{Any},n::Int)
    x,y,photons=getxy(states)
    X=cat(dims=1,x',y')
    R = kmeans(X, n; maxiter=200, display=:iter)
    @assert nclusters(R) == n # verify the number of clusters
    a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers

    state_mapn=StateFlatBg_Results(n)

    for ii=1:n
        x_n=X[1,a.==ii]
        y_n=X[2,a.==ii]
        p_n=photons[a.==ii]
        state_mapn.x[ii]=mean(x_n)    
        state_mapn.y[ii]=mean(y_n)
        state_mapn.photons[ii]=mean(p_n)
        state_mapn.σ_x[ii]=std(x_n) 
        state_mapn.σ_y[ii]=std(y_n) 
        state_mapn.σ_photons[ii]=std(p_n) 
    end
    return state_mapn
end

function getmapn(states::Vector{Any})
    mapnstates,n=getmapnstates(states)
    return clusterstates(mapnstates,n)
end

"""
    getposterior(states::Vector{Any}, sz::Int32, zoom::Int32)

Get a 2D posterior probability image of source locations

# Arguments
- `states::Vector{Any}`: Array of states found by RJMCMC`
- `sz::Int32' : Linear size of analyzed ROI (pixels)
- `zoom::Int32' : Pixel subsampling


"""
function getposterior(states::Vector{Any}, sz::Int, zoom::Int)
    x,y=getxy(states)
    xbins = range(0.5, sz+0.5;step=1 / zoom)
    ybins = range(0.5, sz+0.5;step=1 / zoom)
    h = fit(Histogram,(x,y),(xbins,ybins))
    return h.weights
end


