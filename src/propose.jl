
## State proposal Functions

# Jumptypes are:
# bg, move, add,remove, split, merge


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
    genmodel!(currentstate, rjs, roi)
    genmodel!(teststate, rjs, roitest)
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

