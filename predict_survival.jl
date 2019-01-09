
module predict_survival

using CSV, DataFrames, Mmap, RData, HDF5

include("Choose.jl")
include("compare_model.jl")

const foldFeatures = 75

export meta, rows, cols, mat, train, best, best_v

"ERROR: LoadError: syntax: type declarations on global variables are not yet supported"
function __init__()
    global meta, rows, train, cols, mat, best, best_v
    "Gleason look-up: celfile_name => Gleason Score"
    meta = nothing # for each col
    "for checking, and because there is no way to name rows in Arrays"
    rows = Array{String,1}() # feature rownames
    cols = Array{String,1}() # celfile_name colnames
    mat = nothing # 2D features x celfiles
    train = nothing # cols used for training
    best = nothing # best feature selection
    best_v = nothing # best correlation value (lower is better)
    nothing
end

function restore_state(id)
    global rows, cols, train, best, best_v
    state = nothing;
    try
        h5open(string("run",id,".h5"),"r") do io
            state = read(io,"state")
        end
    catch
        println("cannot open training subset run",id,".h5");
        exit()
    end

    train = state["train"]
    for (k,c) in zip(train,state["train_cols"])
        if cols[k] != c
            println("train col mismatch ",c);
	        throw(DimensionMismatch(String("train col mismatch",c)))
        end
    end
    best_v = state["v"]
    w = state["best_rows"];
    try
        best = convert(Array{Bool,1},state["best"]);
        test_rows = rows[best];
	    for (k,r) in zip(test_rows,w)
            if k != r
                println("best row mismatch ",r);
	            throw(DimensionMismatch(String("best row mismatch ",r)))
            end
        end
    catch
        best = [ x in w for x in rows ]
    end
    choose = max(foldFeatures,size(w,1))
    best = Choose.without_replacement(size(rows,1),choose;keep=best)
end

"Prepare meta by dropping Gleason Scores of 7, and combine exprs that have a score."
function ready_data(dir)
    global meta, rows, cols, mat
    "get lookup celfile_name => Gleason Score"
    meta = CSV.read(joinpath(dir,"gs.csv"))

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

function report(rows,cols=nothing,mat=nothing)
    string("nrows ",size(rows,1),
	   ( cols != nothing ? string(" ncols ",size(cols,1)) : "" ),
	   ( mat != nothing ? string(" mat ",size(mat)) : "" )
	   )
end

"""
Evaluation of model compares predictions for training and test data.
Ideally, training data will be 100% correct. Higher values are better.
"""
function evaluate(report,modeltype, train_features, train_labels, test_features, test_labels; kwargs...)    
compare_model.evaluate(report,modeltype, train_features, train_labels, test_features, test_labels; kwargs...)    
compare_model.evaluate(report,modeltype, test_features, test_labels, train_features, train_labels; kwargs...)    
end

"""
Saved state and one of eight algorithms.
type may be: Bool to use boolean classifiers (GS>7),
	Integer to use cmp(GS,7) (Integers -1 or 1),
	or other to classify using GS values (themselves Integers).
best_rows_file may be: Any to all rows,
	or <:Integer to choose that many at random.
"""
function run(id=1,dir="gene";model=id,type=Bool,best_rows_file=nothing)
    global meta, rows, cols, mat, train, best, best_v
    @time ready_data(dir) ; println("data ready");
    restore_state(id)
    if best_rows_file != nothing if best_rows_file == Any
        choose = size(rows,1)
        best = Choose.without_replacement(size(rows,1),choose);
	println("setting best ",report(best)," from $dir/$best_rows_file")
    elseif typeof(best_rows_file) <:Integer
	choose = max(foldFeatures,best_rows_file);
        best = Choose.without_replacement(size(rows,1),choose);
    else
    try
        w = CSV.read(joinpath(dir,best_rows_file))[2]
	choose = max(foldFeatures,size(w,1));
	best = Choose.without_replacement(size(rows,1),choose;
					  keep=[ x in w for x in rows ])
	println("setting best ",report(best)," from $dir/$best_rows_file")
    catch
        println("cannot set best rows from $dir/$best_rows_file")
	exit()
    end end end

    if type==Bool
	    # cannot handle BitArray{1}
        labels = convert(Array{Bool,1},meta[:gs].>7)
	compare_model.set_label(p->convert(Array{Bool,1},p.>0.5))
    elseif type==Integer
	    # -1 or 1 for GS < 7 or GS > 7
        labels = cmp.(meta[:gs],7)
	compare_model.set_label(p->cmp.(p,0.5))
    else
	    # pairs with (int) classes instead of prob
        labels = convert.(type,meta[:gs])
	compare_model.set_label(p->trunc.(type,p.+0.5))
    end
    keep = ones(Bool,size(cols,1))
    keep[train] .= 0
    others = (1:size(cols,1))[keep]
    train_labels = labels[train]
    train_features = mat[best,train]
    test_labels = labels[others]
    test_features = mat[best,others]
    
    report(rows[best],cols[train],view(mat,best,train))
if model==1
"Does well even with all features."
evaluate(report,compare_model.LinearModel, train_features, train_labels, test_features, test_labels)#;eps=0.01)
elseif model==2
"""
Does Array{Bool,1} yet not Bitvector{1}. Will also handle Integer {-1,1} and Gleason Score classes.
Library code has bugs when using subarrays.
Random result, repeated runs work or dismally fail.
"""
evaluate(report,compare_model.SVM{type}, train_features, train_labels, test_features, test_labels)
elseif model==3
"Library cannot handle Adjoint, so copy arrays."
t1=convert(Array{Float64,2},train_features')
t2=convert(Array{Float64,2},test_features')
evaluate(report,compare_model.DecisionTreeClassifier, t1, train_labels, t2, test_labels)
elseif model==4
"Library cannot handle Adjoint, so copy arrays."
t1=convert(Array{Float64,2},train_features')
t2=convert(Array{Float64,2},test_features')
evaluate(report,compare_model.RandomForestClassifier, t1, train_labels, t2, test_labels)
elseif model==5
"Will not compile."
evaluate(report,compare_model.Booster, train_features, train_labels, test_features, test_labels)
elseif model==6
"Very slow with all features. Benefits by using only best features."
evaluate(report,compare_model.HybridNB, train_features, train_labels, test_features, test_labels)
elseif model==7
"@tf macro is busted for example Colon(...;name=) barfs"
evaluate(report,compare_model.TensorFlowClassifier, train_features, train_labels, test_features, test_labels)
elseif model==8
"Will not compile."
evaluate(report,compare_model.Flux.Chain, train_features, train_labels, test_features, test_labels)
end

end

end # module predict_survival
