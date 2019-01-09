
using DecisionTree
import ScikitLearnBase

function fit!(model::ScikitLearnBase.BaseClassifier, tfeatures, labels)
    #could use ScikitLearnBase.fit!(SVC, tfeatures, Float64.(labels)), but it doesn't take extra args same way.
    ScikitLearnBase.fit!(model, tfeatures, labels)
end

function predict(model::ScikitLearnBase.BaseClassifier, tfeatures)
    ScikitLearnBase.predict_proba(model, tfeatures)[:,2]
end

function fit(kind::Type{<:ScikitLearnBase.BaseClassifier}, tfeatures, labels, args...;kwargs...)
    fit!(kind(args...;kwargs...), tfeatures, labels) #ScikitLearn stores most parameters in the model type
end

