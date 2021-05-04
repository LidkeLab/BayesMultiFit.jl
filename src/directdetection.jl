## This implements data types and methods for direct detection imaging (i.e. standard array imaging at image plane)


abstract type ArrayDD <: BAMFData end

mutable struct ArrayDD_CUDA <: ArrayDD  # direct detection data 
    sz::Int32
    data::CuArray{Float32,2}
end
mutable struct ArrayDD_CPU <: ArrayDD  # direct detection data 
    sz::Int32
    data::Array{Float32,2}
end
ArrayDD_CUDA(sz) = ArrayDD(sz, CuArray{Float32}(undef, sz, sz))
ArrayDD(sz) = ArrayDD_CPU(sz, Array{Float32}(undef, sz, sz))
function deepcopy(a::ArrayDD_CPU)
    b=ArrayDD(a.sz)
    for nn=1:b.sz*b.sz
        b.data[nn]=a.data[nn]
    end
end

function likelihoodratio(m::ArrayDD, mtest::ArrayDD, d::ArrayDD)
    return likelihoodratio(m.data,mtest.data,d.data)
end

function poissrnd(d::ArrayDD)
    out = ArrayDD(d.sz)
    for nn = 1:d.sz^2
        out.data[nn] = Float32(rand(Poisson(Float64(d.data[nn]))))
    end
    return out
end

function genBAMFData(data::ArrayDD)
    return ArrayDD(data.sz)
end



function calcresiduum(model::ArrayDD,data::ArrayDD)
    residuum = ArrayDD(model.sz);
    for nn=1:model.sz*model.sz
        residuum.data[nn]=data.data[nn]-model.data[nn]
    end
    return residuum
end


function makepdf!(a::ArrayDD)
#make all elements sum to 1
mysum=0;
for nn=1:a.sz*a.sz
    a.data[nn]=max(0,a.data[nn])
    mysum+=a.data[nn]
end
for nn=1:a.sz*a.sz
    a.data[nn]/=mysum
end
end

function makecdf!(a::ArrayDD)
#make the array a normalized CDF
makepdf!(a)
for nn=2:a.sz*a.sz
    a.data[nn]+=a.data[nn-1];
end
end

function arrayrand!(a::ArrayDD)
#pull random number from pdf array
#this converts input to cdf
    makecdf!(a)
    r=rand()
    nn=1
    while a.data[nn]<r
        nn+=1
    end
    ii=rem(nn,a.sz)
    jj=ceil(nn/a.sz)
    return ii,jj 
end
    
function arraypdf(a::ArrayDD,ii::Int32,jj::Int32)
#calculate probability at pixel index
mysum=0;
for nn=1:a.sz*a.sz
    mysum+=max(1,a.data[nn]);
end
  return max(1,a.data[ii,jj])/mysum
end