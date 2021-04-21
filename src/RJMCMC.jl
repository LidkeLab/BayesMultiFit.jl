module RJMCMC

include("BAMF.jl")
using .BAMF

burnlength=1000;
chainlength=1000;

mutable struct RJMCMCStruct
    burnin::Int32
    iterations::Int32
    njumptypes::Int32 #number of jump types for RJMCMC
    jumpprobability::Vector{Float32} #sums to 1
    proposalfuns
    acceptfuns
end

mutable struct RJChain  #this is the main output of RJMCMC
    n::Int32  #number of jumps in chain 
    states::Vector{Any}
    jumptypes::Vector{Int32}
    α::Vector{Float32}
    accept::Vector{Bool}
end
RJChain(n::Int32)=RJChain(n,
Vector{Any}(undef,n),
Vector{Int32}(undef,n),
Vector{Float32}(undef,n),
Vector{Bool}(undef,n),
)
function length(rjc::RJChain)
    return rjc.n
end


function initchain(rjs::RJMCMCStruct,burninchain::RJChain) #this gets last state of burnin chain and intializes new chain
    newchain=RJChain(rjs.iterations)
    newchain.states[1]=burninchain.states[burninchain.n]
    newchain.accept[1]=1
    newchain.α[1]=1
    newchain.jumptypes[1]=0
return newchain
end

function initchain(rjs::RJMCMCStruct,intialstate) #this initializes new chain given an intial state and configures a burnin
    njumps=Int32(max(rjs.burnin,1)); #handle zero burn in case
    newchain=RJChain(njumps)
    newchain.states[1]=intialstate
    newchain.accept[1]=1
    newchain.α[1]=1
    newchain.jumptypes[1]=0
return newchain
end

function jtrand(rjs::RJMCMCStruct) #select a jump type
    r=rand(Float32);
    jt=1
    tmp=rjs.jumpprobability[jt]
    while r>tmp
        jt+=1
        if jt==rjs.njumptypes
            break
        end
        tmp=rjs.jumpprobability[jt]
    end    
    return jt
end

function buildchain(rjs::RJMCMCStruct,mhs,intialstate)
    
    #init and burnin    
    bchain=initchain(rjs,intialstate)
    if rjs.burnin>0
        runchain!(rjs,bchain,rjs.burnin,mhs)
    end

    #real chain
    chain=initchain(rjs,bchain)
    runchain!(rjs,chain,rjs.iterations,mhs)

    return chain
end


function runchain!(rjs::RJMCMCStruct,rjc::RJChain,iterations,mhs)
    for nn=1:iterations-1
        jt=jtrand(rjs)
        rjc.jumptypes[nn]=jt;

        #get proposal
        mtest=rjs.proposalfuns[jt](mhs,rjc.states[nn])     

        #calculate acceptance probability
        α=rjs.acceptfuns[jt](mhs,rjc.states[nn],mtest)
        rjc.α[nn+1]=α;

        #update chain
        if α>rand()
            rjc.accept[nn+1]=1;
            rjc.states[nn+1]=mtest;
        else
            rjc.accept[nn+1]=0;
            rjc.states[nn+1]=rjc.states[nn];
        end
       
        #println((nn,jt,α))
    end
end




end

