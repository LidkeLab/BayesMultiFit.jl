## Analysis and Display

function circleShape(h, k, r)
    θ = LinRange(0, 2 * π, 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end


function getxy(states::Vector{Any})
    # count number of emitters 
    nemitters = 0;
    for nn = 1:length(states) 
        nemitters += states[nn].n
    end
    
    x = Vector{Float32}(undef, nemitters)
    y = Vector{Float32}(undef, nemitters)
 
    cnt = 0;
    for ss = 1:length(states) 
        for nn = 1:states[ss].n
            cnt += 1  
            x[cnt] = states[ss].x[nn]
            y[cnt] = states[ss].y[nn]
        end
    end   
    return x,y 
end

function getn(states::Vector{Any})
    maxemitters = 0
    nstates=length(states) 

    traj_n=Vector{Int32}((undef),nstates)
    for nn = 1:nstates
        traj_n[nn]=states[nn].n
        maxemitters=max(maxemitters,states[nn].n)
    end

    posterior_n=zeros(Float32, maxemitters+1) 
    for nn = 1:nstates
        posterior_n[states[nn].n+1]+=states[nn].n
    end
    posterior_n=posterior_n/sum(posterior_n)

    cntmax = maximum(posterior_n)
    coords = findall(x -> x == cntmax, posterior_n)
    map_n=coords[1]-1

    return map_n,posterior_n,traj_n
end

function histogram2D(states::Vector{Any}, sz::Int32, zoom::Int32)
    x,y=getxy(states)
    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    histplot = histogram2d(x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=true) 
    return histplot
end

function histogram2D(states::Vector{Any}, sz::Int32, zoom::Int32, d::ArrayDD, truestate::StateFlatBg)
    x,y=getxy(states)
    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    fig = heatmap(d.data, color=:greys)
    histogram2d!(fig, x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false) 
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue)  
    end
    return fig
end

function histogram2D(states::Vector{Any}, sz::Int32, zoom::Int32, truestate::StateFlatBg)
    x,y=getxy(states)
    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    fig=histogram2d(x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false) 
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue)  
    end
    histogram2d!(fig, x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false,yflip=true) 
    # scatter!(fig, x, y) 
    return fig
end

#"Show model and data vs chain step"
function showoverlay(states::Vector{Any},rjs::RJStruct)
    len=length(states)
    tmp=genBAMFData(rjs)
    sz=rjs.data.sz
    d=Array{Float32}(undef,(sz,sz*size(tmp.data,3),len))
    m=Array{Float32}(undef,(sz,sz*size(tmp.data,3),len))
    println(size(d))
    dataim=reshape(rjs.data.data,(rjs.sz,rjs.sz*size(rjs.data.data,3)))

    println(states[1])
    for nn=1:len
        d[:,:,nn]=dataim     
        genmodel!(states[nn],rjs,tmp)
        m[:,:,nn]=reshape(tmp.data,(rjs.sz,rjs.sz*size(rjs.data.data,3)))
    end

    globmax=maximum((maximum(m),maximum(d)))
    d=d./globmax
    m=m./globmax
    out=cat(dims=1,d,m,d-m)
    println(size(out))
    imshow(out)
end

function plotstate(truestate::StateFlatBg,foundstate::StateFlatBg)
    fig=plot()
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue)  
    end

    for nn = 1:foundstate.n
        plot!(fig, circleShape(foundstate.x[nn], foundstate.y[nn], dcircle), color=:red)  
    end
    
    return fig
end




