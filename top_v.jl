#!/usr/bin/env julia
using HDF5

m = Inf
k = 0
train = nothing
for id in cat(1:7,11:17;dims=1)
    try
        h5open(string("run",id,".h5"),"r") do io
	    global m,k,train
            v = read(io,"state/v")
            println("#",id,"\t",v);
            if v < m
                m = v
                k = id
            end
	    t = read(io,"state/train");
	    if train==nothing || !all(train .== t)
		train = t;
		println(size(train,1),": ",train);
	        r = io["state/best_rows"];
	        println("using ",size(r,1)," features");
            end
            close(io)
        end
    catch
        break
    end
end
println("#",k,"\t",m);
k
