

module Choose

export without_replacement

"""
From is Integer n (the first few positive integers), array or range (to choose from).
Choose == 0 means choose another, keep will have exactly one more true.
Optional already-choosen items to keep flagged in Array{Bool,1}, any size.
Returns Array{Bool,1}, size n or more, with at least choose many true.
If choosing from array or range, returns the result of indexing instead.
"""
function without_replacement(from,choose=0;keep=nothing)
    f = p->p;
    if typeof(from) <:Array || typeof(from) <:UnitRange
	n = size(from,1)
	f = p->from[p]
    else
	n = convert(Integer,from)
    end
	"Assume n >> choose, usually"
    if n > choose
        k = 0
        if keep == nothing keep = zeros(Bool,n)
        else
            k = count(keep)
	    "Pad to at least n"
	    if (m = size(keep,1))<n keep = [ keep; zeros(Bool,n-m) ] end
        end
	"flag one more if zero"
        if choose==0
            choose = k+1
        end
    end
    if n <= choose return f(ones(Bool,n)) end
    "rand() may return same more than once"
    while k < choose
        for i in rand((k<div(n,2) ? (1:n) : (1:n)[.!keep]),choose-k)
            @inbounds if !keep[i] keep[i] = true; k += 1 end
        end
    end
    f(keep)
end

end # module Choose

