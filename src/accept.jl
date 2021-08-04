## Acceptance probability calculations

# Jumptypes are:
# bg, move, add,remove, split, merge


function accept_move(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,idx::Int32)   
    
    if idx<1 return 0 end
    roi = genBAMFData(rjs)
    roitest = genBAMFData(rjs)
    genmodel!(currentstate, rjs, roi)
    genmodel!(teststate, rjs, roitest)
    if (teststate.n>0)&& (minimum(teststate.photons)<0)
        println(currentstate)
        println(teststate)
    end

    LR=likelihoodratio(roi, roitest, rjs.data)
    PR=priorpdf(rjs.prior_photons,teststate.photons[idx])/
        priorpdf(rjs.prior_photons,currentstate.photons[idx])        
    
    α = PR*LR
    return α
end

function accept_bg(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState)

    return rand()
end

function accept_add(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,idx::Int32)
    roi = genBAMFData(rjs)
    roitest = genBAMFData(rjs)
    genmodel!(currentstate, rjs, roi)
    genmodel!(teststate, rjs, roitest)
    LLR = likelihoodratio(roi, roitest, rjs.data)
    #proposal probability
    residuum=calcresiduum(roitest,rjs.data)   
    jj=Int32(min(roi.sz,max(1,round(teststate.x[idx]))))
    ii=Int32(min(roi.sz,max(1,round(teststate.y[idx]))))
    p=arraypdf(residuum,ii,jj)
    α = LLR*(roi.sz+rjs.bndpixels)^2/p
    #println(("add: ",α,ii,jj,teststate.photons[end]))
    return α
end

function accept_remove(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,idx::Int32)
    #use the inverse function
    if idx<1 return 0 end
    α=accept_add(rjs,teststate,currentstate,idx)
    return 1/α
end

function accept_split(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,vararg=(Int32,Int32,Float32,Float32,Float32))

    idx1,idx2,u1,u2,u3=vararg

    if idx1==0
        return 0
    end
    
    roi = genBAMFData(rjs)
    roitest = genBAMFData(rjs)
    genmodel!(currentstate, rjs, roi)
    genmodel!(teststate, rjs, roitest)
    LLR = likelihoodratio(roi, roitest, rjs.data)
    
    #proposal probability
    split_std = rjs.split_std
    N=Normal(0,split_std)
    p=pdf(N,u2)*pdf(N,u3)

    #ratio on intensity priors
    IR=priorpdf(rjs.prior_photons,teststate.photons[idx1])*priorpdf(rjs.prior_photons,teststate.photons[idx2])/
        priorpdf(rjs.prior_photons,currentstate.photons[idx1])  

    #ratio of XY priors 
    XYPR=1/(roi.sz+rjs.bndpixels)^2    
    #Jacobian    
    J=currentstate.photons[idx1]/(1-u1)^2
    PR=LLR*IR*XYPR/p*J
    α = PR/teststate.n 
    #println(("add: ",α,ii,jj,teststate.photons[end]))
    return α

end

function accept_merge(rjs::RJStruct, currentstate::BAMFState, teststate::BAMFState,vararg=(Int32,Int32,Float32,Float32,Float32))
    idx1,idx2,u1,u2,u3=vararg
    if idx1<1 return 0 end
    α=accept_split(rjs,teststate,currentstate,(idx1,idx2,u1,u2,u3))
    return 1/α
end
