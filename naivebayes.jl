
import NaiveBayes: predict_logprobs, HybridNB, restructure_matrix

fit(::Type{HybridNB}, features, labels) = fit(HybridNB(labels), features, labels) 

function predict(model::HybridNB, features)
    unnormed_logprobs = predict_logprobs(model, restructure_matrix(features), Dict{Symbol, Vector{Int}}())
    ps = exp.(unnormed_logprobs)
    (ps./sum(ps,1))[1,:]
end
