## Helper functions
function randID(k::Int32)
    return Int32(ceil(k * rand()))
end

function poissrnd!(d::ArrayDD)
    for nn = 1:d.sz^2
        d.data[nn] = Float32(rand(Poisson(Float64(d.data[nn]))))
    end
end
function poissrnd(d::ArrayDD)
    out = ArrayDD(d.sz)
    for nn = 1:d.sz^2
        out.data[nn] = Float32(rand(Poisson(Float64(d.data[nn]))))
    end
    return out
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

function curandn()
    r=0;
    for ii=1:12
        r+=rand()
    end
    return r-6f0
end