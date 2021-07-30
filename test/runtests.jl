using BayesMultiFit
using Test
BAMF=BayesMultiFit
using ReversibleJumpMCMC
const RJMCMC = ReversibleJumpMCMC
using Random
using Distributions

include("testhelpers.jl")

@testset "BayesMultiFit.jl" begin
    # test blank BAMFData object generation for DD type data for all BAMFData types
    DDdatatypelist=[BAMF.ArrayDD, BAMF.DataSLIVER, BAMF.AdaptData]
    for datatype in DDdatatypelist
        data= genBAMFDD(datatype)
        sz, dim, DD=DDinfo(data)
        @test sz == 32
        if length(dim)==3
            @test dim==(32, 32, 1)
        else
            @test dim==(32,32)
        end
        @test isa(DD, datatype)
    end
    
    # test genBAMFData copy for DD type data for all BAMFData types
    for datatype in DDdatatypelist
        data= genBAMFDD(datatype)
        copy= BAMF.genBAMFData(data)
        sz, dim, DD=DDinfo(copy)
        @test sz == 32
        if length(dim)==3
            @test dim==(32, 32, 1)
        else
            @test dim==(32,32)
        end
        @test isa(DD, datatype)
    end
    
    datastate, psf, xystd,istd,split_std,bndpixels,prior_photons= gendatastate(Int32(1))
    
    # test genmodel! for DD type data for all BAMFData types
    expectDD=BAMF.ArrayDD(Int32(32))
    rjsDD = BAMF.RJStruct(32,psf,xystd,istd,split_std,expectDD,bndpixels,prior_photons)
    BAMF.genmodel!(datastate, rjsDD, expectDD)
    for datatype in DDdatatypelist
        data=genBAMFDD(datatype)
        BAMF.genmodel!(datastate, rjsDD, data)
        sz, dim, DD=DDinfo(data)
        @test sz == 32
        if length(dim)==3
            @test dim==(32, 32, 1)
            for i in 1:32, j in 1:32
                @test data.data[i, j, 1] ≈ expectDD.data[i, j] atol=1f-4
            end
        else
            @test dim==(32,32)
            for i in 1:32, j in 1:32
                @test data.data[i, j] ≈ expectDD.data[i, j] atol=1f-4
            end
        end
        @test isa(DD, datatype)
        
    end
    
    # test blank BAMFData object generation for SLIVER type data for all applicable BAMFData types
    SLIVERdatatypelist=[BAMF.DataSLIVER, BAMF.AdaptData]
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        sz, dim, inttime, invx, invy, SLIVER = SLIVERinfo(data)
        @test sz == 32
        @test dim == (32, 32, 2)
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
    end
    
    # test genBAMFData copy for SLIVER type data for all applicable BAMFData types
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        copy = BAMF.genBAMFData(data)
        sz, dim, inttime, invx, invy, SLIVER = SLIVERinfo(copy)
        @test sz == 32
        @test dim == (32, 32, 2)
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
    end
    
    # test genmodel! for SLIVER type data for all applicable BAMFData types
    expectSLIVER=BAMF.DataSLIVER(Int32(32), [Int32(2)])
    rjsSLIVER = BAMF.RJStruct(Int32(32),psf,xystd,istd,split_std,expectSLIVER,bndpixels,prior_photons)
    BAMF.genmodel!(datastate, rjsSLIVER, expectSLIVER)
    for datatype in SLIVERdatatypelist
        data = genBAMFSLIVER(datatype)
        BAMF.genmodel!(datastate, rjsSLIVER, data)
        sz, dim, inttime, invx, invy, SLIVER = SLIVERinfo(data)
        @test sz == 32
        @test dim == (32, 32, 2)
        @test inttime == 1
        @test invx == 0
        @test invy == 0
        @test isa(SLIVER, datatype)
        for i in 1:32, j in 1:32, k in 1:2
            @test data.data[i, j, k] ≈ expectSLIVER.data[i, j, k] atol=1f-4
        end 
    end
    
    # test deterministic seed for calcintialstate and buildchain 
    datastate, psf,xystd,istd,split_std,bndpixels,prior_photons= gendatastate(Int32(1))
    
    expectDD=BAMF.ArrayDD(Int32(32))
    rjsDD = BAMF.RJStruct(32,psf,xystd,istd,split_std,expectDD,bndpixels,prior_photons)
    BAMF.genmodel!(datastate, rjsDD, expectDD)
    stateDD = BAMF.calcintialstate(rjsDD, Int32(5));
    RJMCMCDD = genRJMCMC();
    
    mychain=RJMCMC.buildchain(RJMCMCDD,rjsDD,stateDD);
    map_n,posterior_n,traj_n=BAMF.getn(mychain.states);
    exp_posterior = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.032546673, 0.054948926, 0.22437479, 0.2593871, 0.14054245, 0.21754138, 0.070658684];
    n_states, map_n=BAMF.getmapnstates(mychain.states);
    @test map_n == 14
    @test length(posterior_n) == 18
    for i in 1:18
        @test posterior_n[i] ≈ exp_posterior[i] atol=0.0000001
    end
    @test length(n_states) == 263
end
