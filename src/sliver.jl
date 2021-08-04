## Defines types and methods for the SLIVER imaging method

# "n: number of data states
# sz: square size of image in pixels
# type: 1 for DD, 2 for SLIVER
# data: array of data of size sz*sz*n where is n_{DD} +2 n_{SLIVER}
# invx: inversion point in x. (n x 1) vector. Ignored for DD
# invy: inversion point in y. (n x 1) vector. Ignored for DD       
# "
mutable struct DataSLIVER <: BAMFData  # SLIVER Data structure
    n::Int32 
    sz::Int32
    type::Vector{Int32}
    inttime::Vector{Float32}
    data::Array{Float32,3}
    invx::Vector{Float32}
    invy::Vector{Float32}
end

function DataSLIVER(sz::Int32, type::Vector{Int32})
    n = length(type)
    nimages = sum(type)
    inttime = ones(Float32, n) / n
    data = Array{Float32}(undef, sz, sz, nimages)
    invx = zeros(Float32, n)
    invy = zeros(Float32, n)
    return DataSLIVER(n, sz, type, inttime, data, invx, invy)
end

# "create a copy of DataSLIVER type with empty data"
function genBAMFData(data::DataSLIVER)
    n = length(data.type)
    nimages = sum(data.type)

    type = zeros(Float32, n)
    inttime = zeros(Float32, n)
    invx = zeros(Float32, n)
    invy = zeros(Float32, n)

    for nn = 1:n
        type[nn] = data.type[nn]
        inttime[nn] = data.inttime[nn]
        invx[nn] = data.invx[nn]
        invy[nn] = data.invy[nn]
    end
    sz = data.sz
    data = Array{Float32}(undef, sz, sz, nimages)
    return DataSLIVER(n, sz, type, inttime, data, invx, invy)
end

function genmodel!(m::StateFlatBg, sz::Int32, psf::PSF_airy2D, model::DataSLIVER)   
    
    cntr = Int32(ceil((sz + 1) / 2))
    for ii = 1:prod(size(model.data))
        model.data[ii] = m.bg;
    end

    # build amplitude images
    image = 1
    for n_meas = 1:model.n 
        for n_emit = 1:m.n
            for jj = 1:sz
                for ii = 1:sz
                    if model.type[n_meas] == 1 # direct detection image
                        r = sqrt((ii - m.y[n_emit])^2 + (jj - m.x[n_emit])^2)
                        model.data[ii,jj,image] += model.inttime[n_meas] * m.photons[n_emit] * airy_amplitude(r, psf.ν)^2 
                    end

                    if model.type[n_meas] == 2 # sliver images
                        r = sqrt((m.y[n_emit] - model.invy[n_meas] - ii + cntr)^2 + (m.x[n_emit] - model.invx[n_meas] - jj + cntr)^2)
                        sym = sqrt(1 / 2f0 * model.inttime[n_meas] * m.photons[n_emit]) * airy_amplitude(r, psf.ν)
                        asym = sqrt(1 / 2f0 * model.inttime[n_meas] * m.photons[n_emit]) * airy_amplitude(r, psf.ν)

                        r = sqrt((-m.y[n_emit] + model.invy[n_meas] - ii + cntr)^2 + (-m.x[n_emit] + model.invx[n_meas] - jj + cntr)^2)
                        sym += sqrt(1 / 2f0 * model.inttime[n_meas] * m.photons[n_emit]) * airy_amplitude(r, psf.ν)
                        asym -= sqrt(1 / 2f0 * model.inttime[n_meas] * m.photons[n_emit]) * airy_amplitude(r, psf.ν)

                        model.data[ii,jj,image] += 0.5f0 * abs(sym)^2
                        model.data[ii,jj,image + 1] += 0.5f0abs(asym)^2
                    end
                end
            end
            
        end
        image += model.type[n_meas]     
    end

end

function genmodel!(currentstate, rjs::RJStruct, data::DataSLIVER)
    genmodel!(currentstate, rjs.sz, rjs.psf, data)
end

function likelihoodratio(m::DataSLIVER, mtest::DataSLIVER, d::DataSLIVER)
    return likelihoodratio(m.data, mtest.data, d.data)
end

# "sliver uses only the direct detection images for residuum and returns a ArrayDD type"
function calcresiduum(model::DataSLIVER, data::DataSLIVER) 
    residuum = ArrayDD(model.sz)
    for nn = 1:prod(size(residuum.data))
        residuum.data[nn] = 0f0
    end

    image = 1
    for n_meas = 1:model.n
        if data.type[n_meas] == 1
            for jj = 1:data.sz
                for ii = 1:data.sz
                     residuum.data[ii,jj] += data.data[ii,jj,image] - model.data[ii,jj,image]
                end 
            end
        end
        image += model.type[n_meas]
    end
    return residuum
end

