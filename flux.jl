
using Flux

function fit(::Type{Flux.Chain}, features, labels; hidden_layers=[256, 256], kwargs...)
    nfeatures = size(features, 1)
    layer_sizes = [nfeatures; hidden_layers; 1]
    layers = map(2:length(layer_sizes)) do ii
        size_below = layer_sizes[ii-1]
        size_above = layer_sizes[ii]
        Dense(size_below, size_above, σ)
    end
    model = Chain(layers..., transpose)
    
    fit!(model, features, labels; kwargs...)
end

"""
    iter_throttle(f, period; default_return=false)
Helper function to only run `f` every `period` times that it is called.
In rounds which it does not run, returns `default_return`
"""
function iter_throttle(f, period; default_return=false)
    round = 0
    function(args...)
        if round==period
            round=0
            f(args...)
        else
            round+=1
            default_return
        end
    end
end
function fit!(model::Flux.Chain, features, labels; silent=true, max_epochs=100_000, conv_atol=0.005, conv_period=5)
    function loss(x, y) # AFAIK Flux loss must *always* be a closure around the model (or the model be global)
        ŷ = model(x)
        -sum(log.(ifelse.(y, ŷ, 1 .- ŷ))) #Binary Cross-Entropy loss
    end
    
    old_loss_o = Inf
    conv_timer = conv_period   
    function stop_cb(ii,loss_val)
        loss_o = loss_val/length(labels) #real loss is the sum, but we want to use the mean
        if loss_o < old_loss_o - conv_atol
            conv_timer = conv_period
        else
            conv_timer-=1
            if conv_timer < 1
                return true
            end
        end
        old_loss_o = loss_o
        return false
    end
    
    log_cb(ii, loss_val) = println("at $ii loss: ", loss_val/length(labels))
    
    opt = ADAM(params(model))
    dataset = Base.Iterators.repeated((features, labels), max_epochs)
    Flux.train!(loss, dataset, opt)
#        cb = Flux.throttle(silent ? (i,l)->() : log_cb, 10)) # every 10 seconds
#        stopping_criteria = iter_throttle(stop_cb, 100), #Every 100 rounds
#    )
    
    model
end
function predict(model::Flux.Chain, features)
    Flux.Tracker.data(model(features))
end
