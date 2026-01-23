using Flux
using CUDA
using LuxCUDA
using Statistics
using ProgressMeter
using BSON

include("src/data.jl")
include("src/preprocess.jl")
include("src/quantizer.jl")
include("src/dataloader.jl")
include("src/model.jl")
include("src/checkpoint.jl")

using .Data
using .Preprocess
using .Quantizer
using .DataLoader
using .Model
using .Checkpoint

const USE_GPU = CUDA.functional()
if USE_GPU
    CUDA.allowscalar(false)
    println("Using GPU: $(CUDA.name(CUDA.device()))")
else
    println("Using CPU")
end

to_device(x) = USE_GPU ? gpu(x) : x

function reconstruction_loss(model, x)
    x_recon = model(x)
    return Flux.mse(x_recon, x)
end

function train_epoch!(model, dataloader, opt_state)
    total_loss = 0.0
    count = 0

    @showprogress for batch in dataloader
        x = to_device(batch)

        loss, grads = Flux.withgradient(model) do m
            reconstruction_loss(m, x)
        end

        if isnan(loss) || isinf(loss)
            continue
        end

        Flux.update!(opt_state, model, grads[1])

        total_loss += loss
        count += 1
    end

    return total_loss / max(count, 1)
end

function validate(model, dataloader)
    total_loss = 0.0
    count = 0

    for batch in dataloader
        x = to_device(batch)
        loss = reconstruction_loss(model, x)
        total_loss += loss
        count += 1
    end

    return total_loss / max(count, 1)
end

function train_from_scratch(; batch_size = 2, num_epochs = 30, learning_rate = 1e-4)
    mel_seq_len = 80 * 1024

    model = Model.create_model(
        in_chans = 1,
        out_chans = 1,
        embed_dim = 256,
        enc_depth = 6,
        dec_depth = 4,
        heads = 8,
        mlp_dim = 1024,
        patch_size = 16,
        max_len = mel_seq_len,
        dropout_rate = 0.1,
    )

    model = to_device(model)

    opt_state = Flux.setup(Adam(learning_rate), model)

    root = @__DIR__
    train_loader = DataLoader.MelDataLoader(root, "train-clean"; batch_size = batch_size)
    val_loader = DataLoader.MelDataLoader(root, "dev-clean"; batch_size = batch_size)

    best_val_loss = Inf

    for epoch = 1:num_epochs
        train_loss = train_epoch!(model, train_loader, opt_state)
        val_loss = validate(model, val_loader)

        println("Epoch $epoch: train=$train_loss val=$val_loss")

        if val_loss < best_val_loss
            best_val_loss = val_loss
            Checkpoint.save_checkpoint(
                "best_model.bson",
                model;
                epoch = epoch,
                loss = best_val_loss,
            )
            model = to_device(model)
        end
    end
end

function train_from_checkpoint(
    checkpoint_path::String;
    batch_size = 2,
    num_epochs = 30,
    learning_rate = 1e-4,
)
    model, metadata = Checkpoint.load_checkpoint(checkpoint_path)
    model = to_device(model)

    opt_state = Flux.setup(Adam(learning_rate), model)

    root = @__DIR__
    train_loader = DataLoader.MelDataLoader(root, "train-clean"; batch_size = batch_size)
    val_loader = DataLoader.MelDataLoader(root, "dev-clean"; batch_size = batch_size)

    println("\nValidating loaded model...")
    initial_val_loss = validate(model, val_loader)
    println("Initial validation loss: $initial_val_loss")
    best_val_loss = initial_val_loss

    println("\nContinuing training for $num_epochs epochs...")

    for epoch = 1:num_epochs
        train_loss = train_epoch!(model, train_loader, opt_state)
        val_loss = validate(model, val_loader)

        println("Epoch $epoch: train=$train_loss val=$val_loss")

        if val_loss < best_val_loss
            best_val_loss = val_loss
            Checkpoint.save_checkpoint(
                "best_model.bson",
                model;
                epoch = epoch,
                loss = best_val_loss,
            )
            model = to_device(model)
        end
    end
end

function parse_arguments(args)
    params = Dict(
        :mode => :scratch,
        :checkpoint => "",
        :batch_size => 2,
        :num_epochs => 30,
        :learning_rate => 1e-4,
    )

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--scratch"
            params[:mode] = :scratch
            i += 1
        elseif arg == "--continue"
            if i + 1 > length(args)
                error("--continue requires a checkpoint path")
            end
            params[:mode] = :continue
            params[:checkpoint] = args[i+1]
            i += 2
        elseif arg == "--batch-size"
            if i + 1 > length(args)
                error("--batch-size requires a value")
            end
            params[:batch_size] = parse(Int, args[i+1])
            i += 2
        elseif arg == "--epochs"
            if i + 1 > length(args)
                error("--epochs requires a value")
            end
            params[:num_epochs] = parse(Int, args[i+1])
            i += 2
        elseif arg == "--lr"
            if i + 1 > length(args)
                error("--lr requires a value")
            end
            params[:learning_rate] = parse(Float64, args[i+1])
            i += 2
        elseif arg == "--help" || arg == "-h"
            print_usage()
            exit(0)
        else
            println("Unknown argument: $arg")
            print_usage()
            exit(1)
        end
    end

    return params
end

function main()
    params = parse_arguments(ARGS)

    println("=" ^ 60)
    println("ViT Neural Audio Coder Training")
    println("=" ^ 60)
    println("Mode: $(params[:mode])")
    println("Batch size: $(params[:batch_size])")
    println("Epochs: $(params[:num_epochs])")
    println("Learning rate: $(params[:learning_rate])")

    if params[:mode] == :continue
        println("Checkpoint: $(params[:checkpoint])")
    end
    println("=" ^ 60)
    println()

    batch_size = params[:batch_size]
    num_epochs = params[:num_epochs]
    learning_rate = params[:learning_rate]

    if params[:mode] == :scratch
        println("Starting training from scratch...\n")
        train_from_scratch(
            batch_size = batch_size,
            num_epochs = num_epochs,
            learning_rate = learning_rate,
        )
    elseif params[:mode] == :continue
        println("Continuing training from checkpoint...\n")
        train_from_checkpoint(
            params[:checkpoint],
            batch_size = batch_size,
            num_epochs = num_epochs,
            learning_rate = learning_rate,
        )
    end

    println("\nTraining completed!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
