## Model Generators for 2D Gaussian PSFs

using SpecialFunctions

"""
    PSF_airy2D <: PSF

Contains the parameter used to calculate an Airy Pattern PSF

The Airy PSF is  

I(r)=ν²/(4π)(2*J₁(ν*r)/(ν*r))²    

where     

ν=πD/(λf)=2*π*nₐ/λ  

!!! note
    The Gaussian approximation is σ = 0.42*π/ν
"""
struct PSF_airy2D <: PSF
    ν::Float32
end


function airy_amplitude(r::Float32,ν::Float32)
    w=r*ν
    w=max(w,1f-5)
    return (4π)^-(1/2) * ν * (2*besselj1(w)/w)
end


function genmodel!(s::StateFlatBg, sz::Int32, psf::PSF_airy2D, model::Array{Float32,2})
    for ii = 1:sz
        for jj = 1:sz
            model[ii,jj] = s.bg
            for nn = 1:s.n
                r=sqrt((ii - s.y[nn])^2+(jj - s.x[nn])^2) 
                model[ii,jj] += s.photons[nn] * airy_amplitude(r,psf.ν)^2 
            end
        end
    end
end


function max2int(psf::PSF_airy2D)
    return 4π/psf.ν^2
end
