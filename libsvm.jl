
import LIBSVM:svmtrain, SVM, svmpredict, SVC, NuSVC, EpsilonSVR

"""
2018.10.04: parameterize model: Bool, Integer, Int8, etc
"""
function fit(::Type{SVM{T}} where T<:Number, features, labels; solver_type=LIBLINEAR.L2R_LR, kwargs...)
    #could use ScikitLearnBase.fit!(SVC, features, Float64.(labels)), but it doesn't take extra args same way.
    svmtrain(features, labels; svmtype=SVC,probability=false, kwargs...)
end

function predict(model::SVM{T} where T<:Number, features)
    classes, probs = svmpredict(model, features)
    probs[1,:]
end

"""
2018.10.05: function classify overrides generic from compare_model.jl
"""
function classify(model::SVM{T} where T<:Number, features)
    classes, probs = svmpredict(model, features)
    classes
end
