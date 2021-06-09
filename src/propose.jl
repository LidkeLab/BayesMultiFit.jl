
## State proposal Functions

# Jumptypes are:
# bg, move, add,remove, split, merge


function propose_move(rjs::RJStruct, currentstate::BAMFState)
    teststate = StateFlatBg(currentstate)
    # get an emitter
    idx = randID(currentstate.n)
    # move the emitter
    move_emitter!(idx, teststate.x, teststate.y, teststate.photons, rjs)
    return teststate, idx
end
function move_emitter!(ID::Int32, x::CuArray, y::CuArray, photons::CuArray, rjs::RJStruct) 
    @cuda move_emitter_CUDA!(ID, x, y, photons, rjs.xy_std, rjs.I_std)
end
function move_emitter_CUDA!(ID::Int32, x, y, photons, xy_std::Float32, i_std::Float32)
    x[ID] += xy_std * curandn()
    y[ID] += xy_std * curandn()
    photons[ID] += i_std * curandn()
    photons[ID] = max(0, photons[ID])
    return nothing
end
function move_emitter!(ID::Int32, x::Vector{Float32}, y::Vector{Float32}, photons::Vector{Float32}, rjs::RJStruct) 
    if ID < 1 return nothing end
    x[ID] += rjs.xy_std * randn()
    y[ID] += rjs.xy_std * randn()

    x[ID]=max(0-rjs.bndpixels,min(rjs.sz+rjs.bndpixels,x[ID]))
    y[ID]=max(0-rjs.bndpixels,min(rjs.sz+rjs.bndpixels,y[ID]))

    photons[ID] += rjs.I_std * randn()
    photons[ID] = max(rjs.prior_photons.θ_start, photons[ID])
    return nothing
end



function propose_bg()
    return rand()
end


## Add and remove -------------

function propose_add(rjs::RJStruct, currentstate::BAMFState)

    teststate = StateFlatBg(currentstate)
    
    # calc residum
    roi = genBAMFData(rjs)
    roitest = genBAMFData(rjs)
    genmodel!(currentstate, rjs, roi)
    genmodel!(teststate, rjs, roitest)
    residuum = calcresiduum(roitest, rjs.data)   
    # pick pixel
    ii, jj = arrayrand!(residuum)

    # pick intensity from prior
    photons = priorrnd(rjs.prior_photons)
    if photons < 0
        println(photons)
    end
    # add new emitter
    addemitter!(teststate, Float32(ii), Float32(jj), photons)

    return teststate, teststate.n
end


function propose_remove(rjs::RJStruct, currentstate::BAMFState)
    teststate = StateFlatBg(currentstate)
    # get an emitter
    idx = randID(currentstate.n)
    if idx > 0
        removeemitter!(teststate, idx)
    end
    return teststate, idx
end

## ---------------------------




function propose_split(rjs::RJStruct, currentstate::BAMFState)
    
    # println("split")

    if currentstate.n<1
        println("attempting to split 0 sources")
    end

    teststate = StateFlatBg(currentstate)
    # get an emitter
    idx = randID(currentstate.n)

    if idx==0
        return teststate,(idx,teststate.n,0f0,0f0,0f0)

    end

    # these are conserved 
    photons_total = teststate.photons[idx]
	mux = teststate.x[idx]
    muy = teststate.y[idx]

    split_std = rjs.split_std
    u1 = rand(Float32)
	u2 = split_std * randn(Float32)
	u3 = split_std * randn(Float32)

	# Change one emitter
    teststate.photons[idx] = u1 * photons_total
    teststate.x[idx] = mux + u2
    teststate.y[idx] = muy + u3

    # add an emitter 
    photons_new = (1 - u1) * photons_total
    x_new = mux - u1 * u2 / (1 - u1)
    y_new = muy - u1 * u3 / (1 - u1)    

    addemitter!(teststate, x_new, y_new, photons_new)

    return teststate,(idx,teststate.n,u1,u2,u3)

end


function propose_merge(rjs::RJStruct, currentstate::BAMFState)
    
    
    # println("merge")
    teststate = StateFlatBg(currentstate)
    if currentstate.n<2 return teststate,(Int32(0),Int32(0),0f1,0f1,0f1) end


    # get an emitter
    idx1 = randID(currentstate.n)
    idx2 = findother(currentstate,idx1)

    #keep the lower idx
    if idx1>idx2
        tmp=idx2
        idx2=idx1
        idx1=tmp
    end

    #merged values
    mux = (currentstate.x[idx1]*currentstate.photons[idx1]+
    currentstate.x[idx2]*currentstate.photons[idx2])/
    (currentstate.photons[idx1]+currentstate.photons[idx2])

    muy = (currentstate.y[idx1]*currentstate.photons[idx1]+
    currentstate.y[idx2]*currentstate.photons[idx2])/
    (currentstate.photons[idx1]+currentstate.photons[idx2])

    photons_total = currentstate.photons[idx1]+currentstate.photons[idx2]

    #replace idx1
    teststate.x[idx1]=mux
    teststate.y[idx1]=muy
    teststate.photons[idx1]=photons_total
    #remove second
    removeemitter!(teststate,idx2)
    
    u1=currentstate.photons[idx1]/photons_total
    u2=currentstate.x[idx1]-currentstate.x[idx2]
    u3=currentstate.y[idx1]-currentstate.y[idx2]

    return teststate,(idx1,idx2,u1,u2,u3)
end

