using BayesMultiFit
using Test
BAMF=BayesMultiFit
using ReversibleJumpMCMC
const RJMCMC = ReversibleJumpMCMC
using Random
using Distributions
using MicroscopePSFs
PSF=MicroscopePSFs

include("testhelpers.jl")

@testset "BayesMultiFit.jl" begin
    
    #TODO: update tests to use examples and test overall functionality
    datastate, psf, xystd,istd,split_std,bndpixels,prior_photons= gendatastate(Int32(1))
    
    # test genmodel! for DD type data for all BAMFData types
    DDdatatypelist=[BAMF.ArrayDD]
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
            #@test isapprox(data.data , expected.DD.data; atol=1f-4)
            #for i in 1:32, j in 1:32
            #    @test data.data[i, j] ≈ expectDD.data[i, j] atol=1f-4
            #end
        end
        @test isa(DD, datatype)
        
    end
end
