## Model Generators for 2D Gaussian PSFs


function genmodel!(s::StateFlatBg, sz::Int, psf::MicroscopePSFs.PSF, 
    model::Array{Float32,2})

    for ii=1:sz^2
        model[ii]=s.bg;
    end

    for jj = 1:sz, ii = 1:sz,nn = 1:s.n
                model[ii+sz*(jj-1)] += s.photons[nn] *
                MicroscopePSFs.pdf(psf,(jj,ii),(s.x[nn],s.y[nn]))
    end
end

