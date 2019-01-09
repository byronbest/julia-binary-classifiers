
"""
Finds random subset of samples for training, balanced in high/low Gleason Score.
Selects foldFeatures gene expressions which are least internally correlated.
Saves execution state and resulting choices and selections in file named in run instance.
"""
module select_training_subset

include("Choose.jl")

const trainOneIn = 20
const medianThreshold = 0.5
const varsThreshold = 0.3
const wilcoxonThreshold = 0.05
const foldFeatures = 250
const correlationAttempts = 1000000

using CSV, DataFrames, Mmap, RData, HDF5
using Random, Statistics, HypothesisTests, MultipleTesting

export meta, rows, cols, mat, train, better_fold, swap_fold, accept, restore_state, save_state

"ERROR: LoadError: syntax: type declarations on global variables are not yet supported"
function __init__()
    global meta, rows, train, cols, mat, medHigh, medLow, gs, best, best_v
    "Gleason look-up: celfile_name => Gleason Score"
    meta = nothing # for each label
    "median of exprs with high Gleason Score"
    medHigh = nothing # for each feature expression
    "median of exprs with low Gleason Score"
    medLow = nothing # for each feature expression
    "used to test if feature implies high Gleason Score"
    gs = nothing # 1.0 for labels with high score
    "for checking, and because there is no way to name rows in Arrays"
    rows = nothing # feature rownames
    cols = Array{String,1}() # celfile_name colnames
    mat = nothing # 2D features x celfiles
    train = nothing # trainOneIn cols used for training
    best = nothing # best feature selection
    best_v = nothing # best correlation value (lower is better)
    nothing
end

function restore_state(id)
    global rows, cols, train, best, best_v
    state = nothing
    try
        h5open(string("run",id,".h5"),"r") do io
            state = read(io,"state")
            close(io)
        end
    catch
        return state
    end
    train = state["train"]
    for (k,c) in zip(train,state["train_cols"])
        if cols[k] != c
            println("train col mismatch ",c);
	    throw(DimensionMismatch(String("train col mismatch",c)))
        end
    end
    return state
end

function save_state(id)
    global rows, cols, train, best, best_v
    h5open(string("run",id,".h5"),"w") do io
        train_cols = cols[train];
        best_rows = rows[best];
        g = g_create(io,"state");
        g["v"] = best_v;
        g["train"] = convert(Array{Int32,1},train);
        g["train_cols"] = train_cols;
        g["best"] = convert(Array{UInt8,1},best);
        g["best_rows"] = best_rows;
        attrs(g)["Description"] = "Saved state";
        close(io)
    end
    best_v
end

"Prepare meta by dropping Gleason Scores of 7, and combine exprs that have a score."
function ready_data(dir)
    global meta, rows, cols, mat
    "get lookup celfile_name => Gleason Score"
    meta=CSV.read(joinpath(dir,"gs.csv"))

    h5open(joinpath(dir,"desc.h5"),"r") do file
        rows = read(file["rows"])
        cols = read(file["cols"])
        close(file)
    end
    nrow = size(rows,1);println(nrow," feature",nrow!=1 ? "s" : "");
    ncol = size(cols,1);println(ncol," celfile",ncol!=1 ? "s" : "");
    if nrow <= foldFeatures
	throw(DimensionMismatch(String("Fewer than ",foldFeatures," features found. This many genes are used for training")))
    end

    open(joinpath(dir,"mat.bin"),"r") do io
        mat = Mmap.mmap(io,Array{Float64,2},(nrow,ncol))
    end
end

"train is initialized to random subset, and modified to balance high and low GS"
function balance_training(again=false)
    global meta, cols, train, gs
    ncols = size(cols,1)
    """
    Find training samples with high and low Gleason Score.
    We know every row has a GS and none are equal to 7.
    """
    subsetHigh = zeros(Bool,ncols); subsetLow = zeros(Bool,ncols)
    high = low = nothing
    sample = 1
    if again || train == nothing
        "Ensure sample size is even, so can balance training."
        sample = div(ncols,trainOneIn); sample += mod(sample,2)
        println(ncols," labels, sampling ",sample);
        train = shuffle(1:ncols)[1:sample]
    end

    "at most these many tries to balance"
    t = trainOneIn; while true
        @inbounds subsetLow[train] .= .!(subsetHigh[train] .= meta[:gs][train].>7);
        @inbounds high = (1:ncols)[subsetHigh];
        @inbounds low = (1:ncols)[subsetLow];
        """
        Should have equal number of high and low.
        """
        fix = size(high,1) - size(low,1)
        println("Skew high vs low: ",fix);
        if (t -= 1) <= 0
            println("Giving up trying to balance.");
            break
        end
        "rand may choose the same more than once"
        if fix < -1
            @inbounds subsetLow[rand(low,-fix)] .= false;
        elseif fix > 1
            @inbounds subsetHigh[rand(high,fix)] .= false;
        else
            println("Good enough!");
            break
        end
        """
        If the above turned the same col off more than once,
        this will restore the correct number.
        """
        keep = Choose.without_replacement(ncols,sample;
            keep=subsetLow .| subsetHigh)
        @inbounds train = (1:ncols)[keep];
    end
    "needed for Wilcoxon test"
    gs = convert.(Float64,subsetHigh) # cols
    high, low
end

"Keep features with either high or low Gleason above background."
function median_threshold()
    global mat, medHigh, medLow
    """
    Adjusts train to balance number of high and low Gleason Scores.
    """
    high, low = balance_training()
    medHigh = [ median(view(mat,i,high)) for i in 1:size(mat,1) ]
    medLow = [ median(view(mat,i,low)) for i in 1:size(mat,1) ]
    "these are features with range of values over the training sample"
    eitherOver = (medHigh .> medianThreshold) .|
        (medLow .> medianThreshold);
    keep(eitherOver)
end

"Keep features with vars over varsThreshold within training samples."
function vars_threshold()
    global mat, train
    m = view(mat,:,train)
    vars_expressed = var(m,dims=2)
    keep(vars_expressed[:,1] .> varsThreshold)
end

"""
Null hypothesis is features correlate with high Gleason Scores.
This rejects the Null hypothesis. Doesn't work very well to choose.
Fold largest median difference.
"""
function wilcoxon_fold()
    global rows, mat, train, medHigh, medLow, gs
    nrow = size(rows,1);
    println("Wilcoxon size ",nrow," feature",nrow!=1 ? "s" : "");
    pvals = zeros(nrow)
    for i in 1:nrow
        pvals[i] = pvalue(SignedRankTest(view(mat,i,train),gs[train]))
    end
    println("Rank test range ",minimum(pvals)," to ",maximum(pvals));
#=
    pvals = min.(1.0,pvals)  # Holm adjust insists 0 <= pval <= 1
    pvals = adjust(pvals, Holm())
    println("Holm adjusted range ",minimum(pvals)," to ",maximum(pvals));
=#
    features = pvals .< wilcoxonThreshold
    if any(features)
        mHigh = view(medHigh,features)
        mLow = view(medLow,features)
        # median fold difference
        fold_diff = (mHigh .- mLow) ./ (mLow .+ 0.75)
        println("fold diff ",minimum(fold_diff)," to ",maximum(fold_diff));

        df = DataFrame(A = fold_diff, B = rows[features])
        df = head(sort(df,rev=true),foldFeatures)
        return df[:B]
    else
        println("no features selected");
    end
    return nothing
end

"Find better fold by choosing at random. May be repeated."
function better_fold(attempt=0)
    global rows, best, best_v
    n = size(rows,1)
    if best == nothing
        best = Choose.without_replacement(n,foldFeatures)
        best_v = correlate(best)
    end
    while attempt > 0
        println(best_v," trying better fold");
        @time for i in 1:correlationAttempts
            s = Choose.without_replacement(n,foldFeatures)
            v = correlate(s)
            if v < best_v
                best_v = v
                best = s
            end
        end
        attempt -= 1
    end
    best_v
end

"""
Assume the number of choices far exceeds the subset chosen.
Unchoose one feature and choose another at random.
Keep the new selection if correlation is lower.
"""
function swap_fold(attempt=0)
    global rows, best, best_v
    if attempt == 0 return best_v end
    n = size(rows,1)
    s = best
    @assert count(s)==foldFeatures "too few features"
    while attempt > 0
        println("#",attempt,"\t",best_v," trying swap fold");
        on, off = count_bits(s,n,foldFeatures)
        shuffle!(on)
        shuffle!(off)
        swapped_one = false
 #   println("best ",size(best,1);
 #   println("nrows ",size(rows,1));
 #   println("on ",size(on,1));
 #   println("off ",size(off,1));
 #   println("mat ",size(mat));
        t = 1
        @time for j in on, k in off
            "skip if already swapped earlier"
            if !s[j] || s[k] continue end
            @inbounds s[j] = false
            @inbounds s[k] = true
            v = correlate(s)
            if v < best_v
                best_v = v
                best = s
                swapped_one = true
 #               yield();
                continue
            end
            "undo if no better"
            s[j] = true
            s[k] = false
            if (t += 1) > correlationAttempts break end
        end
        if !swapped_one
            println("Nothing swapped");
            #break
        end
        attempt -= 1
    end
    best_v
end

function count_bits(bits,n,count)
    on = zeros(Integer,count)
    off = zeros(Integer,n-count)
    j = k = 1
    for i in 1:n
        if j <= count && bits[i]
            @inbounds on[j] = i
            j += 1
        else
            if k <= n-count
	        @inbounds off[k] = i
            end
            k += 1
        end
    end
    on, off
end

"Accept result of best exprs fold."
function accept()
    global best
    keep(best)
end

"keep these rows"
function keep(these::BitArray{1})
    global rows, mat, medHigh, medLow
    rows = rows[these]
    medHigh = medHigh[these]
    medLow = medLow[these]
    mat = view(mat,these,:)
    println("keep these ",size(these,1));
    println("nrows ",size(rows,1));
    println("mat ",size(mat));
    println("gs ",size(gs,1));
end

"Find sum of absolute correlation for training rows in these columns."
function correlate(these)
    global train, mat
    "view crashes when both these and train are boolean"
    C = cor(view(mat,these,train);dims=2)
    "zero diagonal (everything correlates with itself)"
    @inbounds for d in 1:foldFeatures C[d,d] = 0 end
    sum(abs.(C))
end

"Prepare data and attempt better selection."
function run(id,dir="gene";better=0,swap=10)
    global rows, cols, train, best, best_v
    @time ready_data(dir) ; println("data ready");
    state = restore_state(id);
    """
    This will balance training to ensure equal with high and low GS.
    Selects features from each class with median exprs over background noise.
    Further selects features which most vary across the training sample.
    """
    @time median_threshold() ; println("median");
    @time vars_threshold() ; println("vars");
    w = ( state!=nothing ? state["best_rows"] : @time wilcoxon_fold() );
    state = nothing;
    best = Choose.without_replacement(size(rows,1),foldFeatures;
            keep=[ x in w for x in rows ])
    best_v = correlate(best)    
    better_fold(better)
    swap_fold(swap)

    save_state(id)

    best_v
end



end # module select_training_subset


function run()
    p = []
    for i in cat(2:7;dims=1)
        push!(p, @spawn select_training_subset.run(i;better=0,swap=10))
    end
    select_training_subset.run(1;swap=8)
    wait.(p)
end


