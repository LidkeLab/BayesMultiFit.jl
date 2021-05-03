## Model Generators for 2D Gaussian PSFs

using SpecialFunctions

"Airy PSF is ν²/(4π)(2*J1(ν*r)/(ν*r))²
where ν=πD/(λf)=2*π*nₐ/λ"    
struct PSF_airy2D <: PSF
    ν::Float32
end

function genmodel_airy2D!(s::StateFlatBg, sz::Int32, psf::PSF_airy2D, model::Array{Float32,2})
    for ii = 1:sz
        for jj = 1:sz
            model[ii,jj] = s.bg
            for nn = 1:s.n
                r=sqrt((ii - s.y[nn])^2+(jj - s.x[nn])^2)
                w=r*psf.ν
                w=max(w,1f-5)
                model[ii,jj] += s.photons[nn] / (4π) * psf.ν^2 * (2*besselj1(w)/w)^2 
            end
        end
    end
end

function genmodel!(m::StateFlatBg, sz::Int32, psf::PSF_airy2D, model::ArrayDD)
    genmodel_airy2D!(m, sz, psf, model.data)
end

function max2int(psf::PSF_airy2D)
    return 4π/psf.ν^2
end
