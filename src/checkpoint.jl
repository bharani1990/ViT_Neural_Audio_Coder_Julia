module Checkpoint

using Flux
using BSON
using CUDA

export load_checkpoint, save_checkpoint

function load_checkpoint(checkpoint_path::String)
    if !isfile(checkpoint_path)
        error("Checkpoint file not found: $checkpoint_path")
    end

    println("Loading checkpoint from: $checkpoint_path")
    checkpoint = BSON.load(checkpoint_path)

    if haskey(checkpoint, :model_cpu)
        model = checkpoint[:model_cpu]
        println("Loaded model from checkpoint")
    elseif haskey(checkpoint, :model)
        model = checkpoint[:model]
        println("Loaded model from checkpoint")
    else
        error("No model found in checkpoint file. Available keys: $(keys(checkpoint))")
    end

    metadata = Dict{Symbol,Any}()
    
    if haskey(checkpoint, :metadata)
        metadata = checkpoint[:metadata]
        for (k, v) in metadata
            println("  $k: $v")
        end
    else
        if haskey(checkpoint, :epoch)
            metadata[:epoch] = checkpoint[:epoch]
            println("Previous epoch: $(checkpoint[:epoch])")
        end
        if haskey(checkpoint, :loss)
            metadata[:loss] = checkpoint[:loss]
            println("Previous loss: $(checkpoint[:loss])")
        end
    end

    return model, metadata
end

function save_checkpoint(
    filepath::String,
    model;
    epoch = nothing,
    loss = nothing,
    opt_state = nothing,
    bitrate_kbps = nothing,
    latency_ms = nothing,
)
    model_cpu = cpu(model)

    metadata = Dict{Symbol, Any}()
    !isnothing(epoch) && (metadata[:epoch] = epoch)
    !isnothing(loss) && (metadata[:loss] = loss)
    !isnothing(bitrate_kbps) && (metadata[:bitrate_kbps] = bitrate_kbps)
    !isnothing(latency_ms) && (metadata[:latency_ms] = latency_ms)
    
    if !isempty(metadata)
        BSON.@save filepath model_cpu metadata
    else
        BSON.@save filepath model_cpu
    end

    println("Saved checkpoint to $filepath")
end

end
