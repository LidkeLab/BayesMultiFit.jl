## Acceptance probability calculations

# Jumptypes are:
# bg, move, add,remove, split, merge


function accept_move(rjs::RJStructDD, currentstate::StateFlatBg, teststate::StateFlatBg)   
    roi = ArrayDD(rjs.sz)
    roitest = ArrayDD(rjs.sz)
    genmodel!(currentstate, rjs, roi)
    genmodel!(teststate, rjs, roitest)
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
    genmodel!(currentstate, rjs, roi)
    genmodel!(teststate, rjs, roitest)
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
