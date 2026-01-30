using Flux
using CUDA
using LuxCUDA
using Statistics
using ProgressMeter
using BSON

include("src/data.jl")
include("src/preprocess.jl")
include("src/quantizer.jl")
include("src/dataset.jl")
include("src/dataloader.jl")
include("src/model.jl")
include("src/loss.jl")
include("src/train.jl")
include("src/checkpoint.jl")

using .Data
using .Preprocess
using .Quantizer
using .Dataset
using .DataLoader
using .Model
using .Loss
using .Train
using .Checkpoint

const USE_GPU = CUDA.functional()
if USE_GPU
    CUDA.allowscalar(false)
    println("Using GPU: $(CUDA.name(CUDA.device()))")
else
    println("Using CPU")
end

to_device(x) = USE_GPU ? gpu(x) : x

function train_from_scratch(; batch_size = 2, num_epochs = 30, learning_rate = 1e-4)
    root = @__DIR__
    Train.train!(
        epochs = num_epochs,
        batch_size = batch_size,
        lr = Float32(learning_rate),
        root = root,
        save_dir = root,
        sample_rate = 16000,
    )
end

function train_from_checkpoint(
    checkpoint_path::String;
    batch_size = 2,
    num_epochs = 30,
    learning_rate = 1e-4,
)
    println("Note: Checkpoint loading needs to be integrated with Train module")
    println("For now, training from scratch...")
    train_from_scratch(
        batch_size = batch_size,
        num_epochs = num_epochs,
        learning_rate = learning_rate,
    )
end

function parse_arguments(args)
    params = Dict(
        :mode => :scratch,
        :checkpoint => "",
        :batch_size => 8,
        :num_epochs => 30,
        :learning_rate => 5e-4,
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
