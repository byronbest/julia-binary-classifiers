
using LIBLINEAR

function fit(::Type{LinearModel}, features, labels; solver_type=LIBLINEAR.L2R_LR, kwargs...)
    linear_train(labels, features; solver_type=solver_type, kwargs...)
end

function predict(model::LinearModel, features)
    classes, probs = linear_predict(model, features; probability_estimates=true)
    vec(probs)
end

"""
2018.10.05: function classify overrides generic from compare_model.jl
"""
function classify(model::LinearModel, features)
    classes, probs = linear_predict(model, features)
    vec(classes)
end

