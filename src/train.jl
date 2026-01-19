module Train

using ..Model
using ..Dataset
using ..DataLoader
using Flux
using Functors: @functor
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

struct PatchDecoder
    linear::Dense
end

@functor PatchDecoder

PatchDecoder(embed_dim::Int, patch_dim::Int) =
    PatchDecoder(Dense(embed_dim, patch_dim))

function (d::PatchDecoder)(y)
    E, T, B = size(y)
    y2 = reshape(y, E, T * B)
    out = d.linear(y2)
    out = reshape(out, size(out,1), T, B)
end

function batch_loss(model, batch)
    T, P, B = size(batch)
    recon = model(batch)
    sum((recon .- batch).^2) / (T * P * B)
end

function compute_validation_loss(model, loader)
    total_loss = 0.0f0
    nbatches = 0
    state = 1
    while true
        iter = iterate(loader, state)
        iter === nothing && break
        batch, next_state = iter
        state = next_state
        batch = move_batch(batch)
        total_loss += batch_loss(model, batch)
        nbatches += 1
    end
    total_loss / max(nbatches, 1)
end

function train!(; epochs::Int=1,
                batch_size::Int=4,
                lr::Float32=1f-4,
                root::AbstractString=pwd(),
                save_dir::AbstractString="")

    ds_train = Dataset.AudioDataset(root; split="train-clean")
    loader_train = DataLoader.AudioDataLoader(ds_train; batch_size=batch_size)

    ds_val = Dataset.AudioDataset(root; split="dev-clean")
    loader_val = DataLoader.AudioDataLoader(ds_val; batch_size=batch_size)

    first_batch, _ = iterate(loader_train)
    T, P, _ = size(first_batch)

    encoder = Model.ViTEncoder(
        P,
        Model.EMBED_DIM,
        Model.NUM_LAYERS,
        Model.NUM_HEADS,
        Model.MLP_DIM,
        T
    )
    decoder = PatchDecoder(Model.EMBED_DIM, P)

    encoder = move_model(encoder)
    decoder = move_model(decoder)
    model = Flux.Chain(encoder, decoder)

    opt_state = Flux.setup(Flux.Adam(lr), model)

    if save_dir != ""
        mkpath(save_dir)
    end

    num_samples = length(ds_train)
    total_batches = ceil(Int, num_samples / batch_size)
    best_loss = Inf
    losses_dict = Dict()

    println("Training on $(device_name())")
    println("Num-Epochs = $(epochs), Num-Batch-Size = $(batch_size), Num-Batches-per-epoch = $(total_batches)")

    for epoch in 1:epochs
        loader = DataLoader.AudioDataLoader(ds_train; batch_size=batch_size)
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
                batch_loss(m, batch)
            end

            Flux.update!(opt_state, model, grads[1])

            total_loss += loss
            nbatches += 1
        end

        avg_loss = total_loss / max(nbatches, 1)
        val_loss = compute_validation_loss(model, loader_val)
        losses_dict[epoch] = Dict("train" => avg_loss, "val" => val_loss)

        println("epoch = $epoch, batches = $nbatches, avg loss = $(avg_loss), val loss = $(val_loss)")

        if save_dir != "" && avg_loss < best_loss
            best_loss = avg_loss
            model_path = joinpath(save_dir, "best_model.jld2")
            println("Saving best model to $model_path")
            jldsave(model_path; encoder=encoder, decoder=decoder, epoch=epoch, loss=best_loss)
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
