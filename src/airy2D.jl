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

#midpoint rule approximation to besselj1 integral
function mybesselj1(x::Float32,m::Int32=Int32(10))
    j1=0f0;
    for k=1:(m-1)
        j1+=sin( x *sin(pi/(2.0f0*m)*(k+0.5f0))) * sin(pi/(2.0f0*m)*(k+.5f0))
    end
    return j1/m
end

function airy_amplitude(r::Float32,ν::Float32)
    w=r*ν
    w=max(w,1f-5)
    return (4π)^-(1/2) * ν * (2*mybesselj1(w)/w)
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
