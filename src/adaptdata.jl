#=
MeasType is the abstract type that would hold the information unique to the
measurement type that it identifies
=#

abstract type MeasType end

#=
DDMeasType describes the information necessary to create a DD measurement. This is just the
integration time and the number of images the DD measurement creates
=#

struct DDMeasType <: MeasType
    images::Int32
    inttime::Float32
end

function DDMeasType(inttime::Float32)
    return DDMeasType(Int32(1), inttime)
end

function DDMeasType(info::Tuple, inttime::Float32)
    return DDMeasType(inttime)
end

#=
SLIVERMeasType describes the information necessary to create a SLIVER measurement. That would be the
integration time, the xy coordinates of the inversion point, and the number of images a SLIVER 
measurement creates
=#

struct SLIVERMeasType <: MeasType
    images::Int32
    inttime::Float32
    invx::Float32
    invy::Float32
end

function SLIVERMeasType(inttime::Float32, invx::Float32, invy::Float32)
    return SLIVERMeasType(Int32(2), inttime, invx, invy)
end

function SLIVERMeasType(info::Tuple, inttime::Float32)
    invx, invy = info
    return SLIVERMeasType(inttime, invx, invy)
end

#=
genMeasType generates a MeasType object of the appropriate type for the inputs
=#

function genMeasType(typeinfo::Tuple{DataType, Tuple, Float32})
    modeltype, inputs, inttime = typeinfo
    return modeltype(inputs, inttime)
end

#=
genMeasTypelist(typelist::Vector{MeasTypeTuple}) takes a vector of tuples with structure
(::Type{MeasType}, ::Tuple{Vararg{Float32}, ::Float32}), where the varargs tuple is the non
integration time info and the Float32 entry is integration time, or
(::Type{MeasType}, ::Tuple{Vararg{Float32}), where the integration time is left out and is 
defaulted to be 1/length(typelist)
=#

function genMeasTypelist(typelist::Vector{Tuple{DataType, T}}) where {T<:Tuple}
    n = length(typelist)
    typelist_c = [(entry1, entry2, Float32(1/n)) for (entry1, entry2) in typelist]
    return genMeasTypelist(typelist_c)
end

function genMeasTypelist(typelist::Vector{Tuple{DataType, T, Float32}}) where {T<:Tuple}
    return [genMeasType(meastype) for meastype in typelist]
end

#=
AdaptData is of the BAMFData type and is initialized with the list of measurement types it is meant 
to model.
=#

struct AdaptData <: BAMFData
    n::Int32
    sz::Int32
    meastypes::Vector{MeasType}
    data::Array{Float32, 3}
end

function AdaptData(sz::Int32, meastypes::Vector{T}) where {T<:MeasType}
    n = length(meastypes)
    nimages = sum([meastype.images for meastype in meastypes])
    data = zeros(Float32, sz, sz, nimages)
    return AdaptData(n, sz, meastypes, data)
end

function AdaptData(sz::Int32, meastypes::Vector{T}) where {T<:Tuple}
    meastypelist = genMeasTypelist(meastypes)
    return AdaptData(sz, meastypelist)
end

#=
genBAMFData creates a copy of the BAMFData type passed in, with an empty data.
=#

function genBAMFData(data::AdaptData)
    return AdaptData(data.sz, data.meastypes)
end

#=
spread(psf::PSF, r::Float32) returns the appropriate modifier for the intensity based on distance
from an emitter and the point source function.
=#

function spread(psf::PSF_gauss2D, r::Float32)
    return exp(-(r^2) / (2 * psf.σ^2)) / (2 * π * psf.σ^2)
end

function spread(psf::PSF_airy2D, r::Float32)
    return airy_amplitude(r, psf.ν)
end

#=
genimage(s::BAMFState, sz::Int32, psf::PSF, meas::MeasType) creates an array depending on the measurement
type, the state, the size, and the point spread function.
=#

function genimage(s::BAMFState, sz::Int32, psf::PSF, meas::DDMeasType)
    image = zeros(Float32, sz, sz, 1)
    image += [s.bg for ii in 1:sz, jj in 1:sz, kk in 1:1]
    for n_emit in 1:s.n, ii in 1:sz, jj in 1:sz
        r = sqrt((jj - s.x[n_emit])^2 + (ii - s.y[n_emit])^2)
        image[ii, jj, 1] += s.photons[n_emit]*meas.inttime*(spread(psf, r)^2)
    end
    return image
end

function genimage(s::BAMFState, sz::Int32, psf::PSF, meas::SLIVERMeasType)
    image = zeros(Float32, sz, sz, 2)
    cntr = Int32(ceil((sz + 1) / 2)) 
    for n_emit in 1:s.n, ii in 1:sz, jj in 1:sz
        r1 = sqrt((s.y[n_emit] - meas.invy - ii + cntr)^2 + (s.x[n_emit] - meas.invx - jj + cntr)^2)
        a1 = sqrt(1 / 2f0 * meas.inttime * s.photons[n_emit]) * spread(psf, r1)
        r2 = sqrt((-s.y[n_emit] + meas.invy - ii + cntr)^2 + (-s.x[n_emit] + meas.invx - jj + cntr)^2)
        a2 = sqrt(1 / 2f0 * meas.inttime * s.photons[n_emit]) * spread(psf, r2)
        image[ii, jj, 1] += 0.5f0*abs(a1 + a2)^2
        image[ii, jj, 2] += 0.5f0*abs(a1 - a2)^2
    end
    return image
end

#=
genmodel!(s::BAMFState, sz::Int32, psf::PSF, model::AdaptData) modifies AdaptData to show the appropriate 
images for the measurement types selected, the point spread function, the size, and the state.
=#

function genmodel!(s::BAMFState, rjs::RJStruct, model::AdaptData)
    genmodel!(s, rjs.sz, rjs.psf, model)
end

function genmodel!(s::BAMFState, sz::Int32, psf::PSF, model::AdaptData)
    image = 1
    for meastype in model.meastypes
        images = genimage(s, sz, psf, meastype)
        model.data[:, :, image:image+meastype.images-1] = images
        image+=meastype.images
    end
end
    

    
    