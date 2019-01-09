
import XGBoost: xgboost, Booster

function fit(::Type{Booster}, features, labels; num_rounds=16, eta=1, max_depth = 16, kwargs...)
    xgboost(features', num_rounds; label=labels, objective = "binary:logistic",  eta=eta, max_depth=max_depth, silent=true, kwargs...)
end

function predict(model::Booster, features)
    XGBoost.predict(model, features')
end
