
"""
https://white.ucc.asn.au/2017/12/18/7-Binary-Classifier-Libraries-in-Julia.html

Note Pkg checkout is now Pkg develop
I had to use Pkg rm, then e.g. rm -rf ~/.julia/packages/LIBSVM to actually use dev version.

2018.10.03: Change evaluate to not use globals.
2018.10.05: Change classify to use label function. Overridden by some models, e.g. liblinear.jl
"""

module compare_model

using Printf, Random, Statistics, HypothesisTests, MultipleTesting
import StatsBase: fit!, fit, predict

export set_label

set_label(fn) = label = fn

label = p->p.>0.5 # label probabilities to compare models
classify(model, features) = label(predict(model, features))
accuracy(model, features, ground_truth_labels) = mean(classify(model, features) .== ground_truth_labels)

percent(x) = @sprintf("%0.2f%%", 100*x)

function evaluate(report,modeltype, train_features, train_labels, test_features, test_labels; kwargs...)    
    println(report(train_features,train_labels));
    @time model = fit(modeltype, train_features, train_labels; kwargs...)
    
    println("$modeltype Train accuracy: ", percent(accuracy(model, train_features, train_labels)))
    println("$modeltype Test accuracy: ", percent(accuracy(model, test_features, test_labels)))
end

include("liblinear.jl")

include("libsvm.jl")

include("decisiontree.jl")

#include("xgboost.jl")

include("naivebayes.jl")

include("tensorflow.jl")

include("flux.jl")

end # module compare_model
