## Acceptance probability calculations

# Jumptypes are:
# bg, move, add,remove, split, merge


function checkbounds(rjs::RJStruct, teststate::BAMFState,idx::Int)
    outofbounds=false;
    if (teststate.x[idx] < -rjs.bndpixels)||(teststate.y[idx] < -rjs.bndpixels)||
        (teststate.x[idx] > rjs.sz+rjs.bndpixels)||(teststate.y[idx] > rjs.sz+rjs.bndpixels)
        outofbounds=true;    
    end
    return outofbounds
end


function accept_move(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,idx::Int)   
    
    if idx<1 return 0 end
    genmodel!(currentstate, rjs, rjs.modeldata)
    genmodel!(teststate, rjs, rjs.testdata)
    if (teststate.n>0)&& (minimum(teststate.photons)<0)
        println(currentstate)
        println(teststate)
    end

    LR=likelihoodratio(rjs.modeldata, rjs.testdata, rjs.data)
    PR=priorpdf(rjs.prior_photons,teststate.photons[idx])/
        priorpdf(rjs.prior_photons,currentstate.photons[idx])        
    
    α = PR*LR
    return α
end

function accept_bg(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState)

    return rand()
end

function accept_add(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,idx::Int)
    genmodel!(currentstate, rjs, rjs.modeldata)
    genmodel!(teststate, rjs, rjs.testdata)
    LLR=likelihoodratio(rjs.modeldata, rjs.testdata, rjs.data)
    #proposal probability
    residuum=calcresiduum(rjs.testdata,rjs.data)   
    jj=Int(min(rjs.data.sz,max(1,round(teststate.x[idx]))))
    ii=Int(min(rjs.data.sz,max(1,round(teststate.y[idx]))))
    p=arraypdf(residuum,ii,jj)
    α = LLR*(rjs.data.sz+rjs.bndpixels)^2/p
    #println(("add: ",α,ii,jj,teststate.photons[end]))
    return α
end

function accept_remove(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,idx::Int)
    #use the inverse function
    if idx<1 return 0 end
    α=accept_add(rjs,teststate,currentstate,idx)
    return 1/α
end

function accept_split(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,vararg=(Int,Int,Float32,Float32,Float32))

    idx1,idx2,u1,u2,u3=vararg

    if idx1==0
        return 0
    end

    # reject if split gives value outside x,y prior
    if checkbounds(rjs,teststate,idx1)
        return 0
    end

    if checkbounds(rjs,teststate,idx2)
        return 0
    end

    genmodel!(currentstate, rjs, rjs.modeldata)
    genmodel!(teststate, rjs, rjs.testdata)
    LLR=likelihoodratio(rjs.modeldata, rjs.testdata, rjs.data)
    
    #proposal probability
    split_std = rjs.split_std
    N=Normal(0,split_std)
    p=pdf(N,u2)*pdf(N,u3)

    #ratio on intensity priors
    IR=priorpdf(rjs.prior_photons,teststate.photons[idx1])*priorpdf(rjs.prior_photons,teststate.photons[idx2])/
        priorpdf(rjs.prior_photons,currentstate.photons[idx1])  

    #ratio of XY priors 
    XYPR=1/(rjs.data.sz+rjs.bndpixels)^2    
    #Jacobian    
    J=currentstate.photons[idx1]/(1-u1)^2
    PR=LLR*IR*XYPR/p*J
    α = PR/teststate.n 
    #println(("add: ",α,ii,jj,teststate.photons[end]))
    return α

end

function accept_merge(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,vararg=(Int,Int,Float32,Float32,Float32))
    idx1,idx2,u1,u2,u3=vararg
    if idx1<1 return 0 end
    α=accept_split(rjs,teststate,currentstate,(idx1,idx2,u1,u2,u3))
    return 1/α
end
