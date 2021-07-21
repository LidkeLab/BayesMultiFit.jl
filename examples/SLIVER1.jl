using BayesMultiFit
BAMF = BayesMultiFit

struct SLIVER1DMeasType <: BAMF.MeasType
    images::Int32
    inttime::Float32
    invx::Float32
end

function SLIVER1DMeasType(inttime::Float32, invx::Float32)
    return SLIVER1DMeasType(Int32(2), inttime, invx)
end

function SLIVER1DMeasType(info::Tuple, inttime::Float32)
    invx, = info
    return SLIVER1DMeasType(inttime, invx)
end

function BAMF.genimage(s::BAMF.BAMFState, sz::Int32, psf::BAMF.PSF, meas::SLIVER1DMeasType)
    image = zeros(Float32, sz, sz, 2)
    cntr = Int32(ceil((sz + 1) / 2)) 
    for n_emit in 1:s.n, ii in 1:sz, jj in 1:sz
        r1 = sqrt((s.y[n_emit] - ii)^2 + (s.x[n_emit] - meas.invx - jj + cntr)^2)
        a1 = sqrt(1 / 2f0 * meas.inttime * s.photons[n_emit]) * BAMF.spread(psf, r1)
        r2 = sqrt((-s.y[n_emit] + ii)^2 + (-s.x[n_emit] + meas.invx - jj + cntr)^2)
        a2 = sqrt(1 / 2f0 * meas.inttime * s.photons[n_emit]) * BAMF.spread(psf, r2)
        image[ii, jj, 1] += 0.5f0*abs(a1 + a2)^2
        image[ii, jj, 2] += 0.5f0*abs(a1 - a2)^2
    end
    return image
end