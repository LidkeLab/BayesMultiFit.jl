## Model Generators for 2D Gaussian PSFs


    
struct PSF_gauss2D <: PSF
    σ::Float32
end

function genmodel!(s::StateFlatBg, sz::Int32, psf::PSF_gauss2D, model::Array{Float32,2})
    for ii = 1:sz
        for jj = 1:sz
            model[ii,jj] = s.bg
            for nn = 1:s.n
                model[ii,jj] += s.photons[nn] / (2 * π * psf.σ^2) *
                exp(-(ii - s.y[nn])^2 / (2 * psf.σ^2)) *
                exp(-(jj - s.x[nn])^2 / (2 * psf.σ^2))
            end
        end
    end
end

function genmodel_gauss2D_CUDA!(s_n::Int32,s_x, s_y, 
    s_photons,s_bg::Float32, sz::Int32, σ::Float32, model)
     #note that the 2d array is linearized and using 1-based indexing in kernel
     ii = blockIdx().x
     jj = threadIdx().x 
     
     idx=(ii-1)*sz+jj
     model[idx] = s_bg + 1f-4
     for nn = 1:s_n
         model[idx] += s_photons[nn] / (2 * π * σ^2) *
                 exp(-(ii - s_y[nn])^2 / (2 * σ^2)) *
                 exp(-(jj - s_x[nn])^2 / (2 * σ^2))
     end
     return nothing
end

function genmodel!(s::StateFlatBg, sz::Int32, psf::PSF_gauss2D, model::CuArray{Float32,2}) 
    @cuda threads=sz blocks=sz genmodel_gauss2D_CUDA!(s.n,s.x,s.y,s.photons,s.bg, sz, psf.σ, model)
end

function max2int(psf::PSF_gauss2D)
    return 2*pi*psf.σ
end
