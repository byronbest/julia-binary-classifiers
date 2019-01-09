
using TensorFlow

struct TensorFlowClassifier
    xs::Tensor{Float32}
    ys::Tensor{Float32}
    ys_targets::Tensor{Float32}
    mean_loss::Tensor{Float32}
    optimizer::Tensor # weirdly this is not type-stable, probably a bug I need to chase down. It *should* always be `Tensor{Any}`

    sess::Session
end

function TensorFlowClassifier(nfeatures, hidden_layers)
    layer_sizes = [nfeatures; hidden_layers; 1]
    sess = Session(Graph())
    @tf begin
        xs = placeholder(Float32, shape=[nfeatures, -1]) #Nfeatures, Nsamples (going to use julian ordering)
        zs = xs
# 2018.10.06: @tf doesn't like for loop
#        for ii in 2:length(layer_sizes)
	ii = 2
	while ii <= length(layer_sizes)
            size_below = layer_sizes[ii-1]
            size_above = layer_sizes[ii]
            W = get_variable("W_$ii", [size_above, size_below], Float32)
            b = get_variable("b_$ii", [size_above, 1], Float32)
            zs = nn.sigmoid(W*zs + b; name="zs_$ii")
	    ii += 1
        end
 
# 2018.10.06: squeeze is called dropdims since 1.0
	ys = dropdims(zs; dims=[1]) #drop first dimension
        
        ys_targets = placeholder(Float32, shape=[-1])
        loss = nn.sigmoid_cross_entropy_with_logits(;logits=ys, targets=ys_targets)
        mean_loss = mean(loss)
        optimizer = TensorFlow.train.minimize(TensorFlow.train.AdamOptimizer(), loss)
    end 
    run(sess, global_variables_initializer())
    
    TensorFlowClassifier(xs,ys, ys_targets, mean_loss, optimizer, sess)
end

function fit(::Type{TensorFlowClassifier}, features, labels; hidden_layers=[256, 256], kwargs...)
    nfeatures = size(features, 1)
    model = TensorFlowClassifier(nfeatures, hidden_layers)
    fit!(model, features, labels; kwargs...)
end

function fit!(model::TensorFlowClassifier, features, labels; silent=true, max_epochs=100_000, conv_atol=0.005, conv_period=5)
    conv_timer = conv_period
    old_loss_o = Inf
    for ii in 1:max_epochs
        loss_o, _ = run(model.sess, (model.mean_loss, model.optimizer), Dict(model.xs=>features, model.ys_targets=>labels))
        if ii%1_000==1 
            !silent && println(loss_o)

            # Check for convergence
            if loss_o < old_loss_o - conv_atol
                conv_timer = conv_period
            else
                conv_timer-=1
                if conv_timer < 1
                    break
                end
            end
        end
        old_loss_o = loss_o
    end
    model
end
function predict(model::TensorFlowClassifier, features)
    run(model.sess, model.ys, Dict(model.xs=>features)) 
end
