module Train

using ..Model
using ..DataLoader
using ..Loss
using Flux
using JLD2
using JSON
using CUDA

const GPU_AVAILABLE = try
    CUDA.has_cuda()
catch
    false
end

const move_model = GPU_AVAILABLE ? Flux.gpu : identity
const move_batch = GPU_AVAILABLE ? CUDA.CuArray : identity

device_name() = GPU_AVAILABLE ? "GPU $(CUDA.device())" : "CPU"

function batch_loss(model, batch; sample_rate::Int=16000)
    T, P, B = size(batch)
    recon = model(batch)
    
    loss = Loss.composite_loss(
        recon, 
        batch, 
        sample_rate;
        w_mse = 1.0f0,
        w_spectral = 0.1f0,
        w_correlation = 0.1f0
    )
    
    return loss
end

function compute_validation_loss(model, loader; sample_rate::Int=16000)
    total_loss = 0.0f0
    nbatches = 0
    state = 1
    while true
        iter = iterate(loader, state)
        iter === nothing && break
        batch, next_state = iter
        state = next_state
        batch = move_batch(batch)
        total_loss += batch_loss(model, batch; sample_rate=sample_rate)
        nbatches += 1
    end
    total_loss / max(nbatches, 1)
end

function train!(;
    epochs::Int = 1,
    batch_size::Int = 4,
    lr::Float32 = 1.0f-4,
    root::AbstractString = pwd(),
    save_dir::AbstractString = "",
    sample_rate::Int = 16000,
)
    target_length = 16000

    loader_val = DataLoader.MelDataLoader(root, "dev-clean"; batch_size = batch_size, target_length = target_length)

    model = Model.create_model()
    model = move_model(model)

    opt_state = Flux.setup(Flux.Adam(lr), model)

    if save_dir != ""
        mkpath(save_dir)
    end

    best_loss = Inf
    losses_dict = Dict()

    println("Training on $(device_name())")
    println("Num-Epochs = $(epochs), Num-Batch-Size = $(batch_size)")

    for epoch = 1:epochs
        loader = DataLoader.MelDataLoader(root, "train-clean"; batch_size = batch_size, target_length = target_length)
        total_loss = 0.0f0
        nbatches = 0
        state = 1

        while true
            iter = iterate(loader, state)
            iter === nothing && break
            batch, next_state = iter
            state = next_state

            batch = move_batch(batch)

            loss, grads = Flux.withgradient(model) do m
                batch_loss(m, batch; sample_rate=sample_rate)
            end

            Flux.update!(opt_state, model, grads[1])

            total_loss += loss
            nbatches += 1
        end

        avg_loss = total_loss / max(nbatches, 1)
        val_loss = compute_validation_loss(model, loader_val; sample_rate=sample_rate)
        losses_dict[epoch] = Dict("train" => avg_loss, "val" => val_loss)

        println("epoch = $epoch, batches = $nbatches, avg loss = $(avg_loss), val loss = $(val_loss)")

        if save_dir != "" && avg_loss < best_loss
            best_loss = avg_loss
            model_path = joinpath(save_dir, "best_model.bson")
            println("Saving best model to $model_path")
            jldsave(model_path; model = model, epoch = epoch, loss = best_loss)
        end
    end

    if save_dir != ""
        losses_path = joinpath(save_dir, "losses.json")
        open(losses_path, "w") do f
            JSON.print(f, losses_dict)
        end
        println("Saving losses to $losses_path")
    end

    return model, losses_dict
end

end
