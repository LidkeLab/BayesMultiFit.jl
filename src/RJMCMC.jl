module RJMCMC

burnlength=1000;
chainlength=1000;

mutable struct rjstruct
    burnin::Int32
    iterations::Int32
    njumptypes::Int32 #number of jump types for RJMCMC
    jumpprobability::Vector{Float32} #sums to 1

end

mutable struct rjstate
    n::Int32
    x::Vector{Float32}
    y::Vector{Float32}
    photons::Vector{Float32}
end
rjstate()=rjstate(1,[0],[0],[0]) #outer constructor

mutable struct rjchain  #this is the main output of RJMCMC
    n::Int32  #number of jumps in chain 
    states::Vector{rjstate}
    jumptypes::Vector{Int32}
    alpha::Vector{Float32}
    accept::Vector{Bool}
end
rjchain(n::Int32)=rjchain(n,
Vector{rjstate}(undef,n),
Vector{Int32}(undef,n),
Vector{Float32}(undef,n),
Vector{Bool}(undef,n),
)

function initstate(rjs::rjstruct) #use priors to get initial state
    n=3 #get this from prior on number
    x=zeros(Float32,n)
    y=zeros(Float32,n)
    photons=zeros(Float32,n)
    for ii=1:n
        x[ii]=0
        y[ii]=0
        photons[ii]=100
    end
    return rjstate(n,x,y,photons)
end


function initchain(rjs::rjstruct,burninchain::rjchain) #this gets last state of burnin chain and intializes new chain
    newchain=rjchain(rjs.iterations)
    newchain.states[1]=burninchain.states[burninchain.n]
    newchain.accept[1]=1
    newchain.alpha[1]=1
    newchain.jumptypes[1]=0
return newchain
end

function initchain(rjs::rjstruct) #this initializes new chain from priors and uses burnin length
    newchain=rjchain(rjs.burnin)
    newchain.states[1]=initstate(rjs)
    newchain.accept[1]=1
    newchain.alpha[1]=1
    newchain.jumptypes[1]=0
return newchain
end

function jtrand(cs::rjstruct) #select a jump type
    r=rand(Float32);
    jt=1
    tmp=cs.jumpprobability[jt]
    while r>tmp
        jt+=1
        if jt==cs.njumptypes
            break
        end
        tmp=cs.jumpprobability[jt]
    end    
    return jt
end

function buildchain(rjs::rjstruct)
    
    #init and burnin
    bchain=initchain(rjs)
    runchain!(rjs,bchain,rjs.burnin)

    #real chain
    chain=initchain(rjs,bchain)
    runchain!(rjs,chain,rjs.iterations)

    return chain
end


function runchain!(rjs::rjstruct,rjc::rjchain,iterations)
    for nn=1:iterations
        jt=jtrand(rjs)

        rjc.accept[nn]=1;
        rjc.jumptypes[nn]=jt;
        rjc.alpha[nn]=1;
        rjc.states[nn]=rjstate();

        println((nn,jt))
    end
end


end

