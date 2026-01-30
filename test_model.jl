using Flux
using CUDA
using LuxCUDA
using BSON

include("src/data.jl")
include("src/preprocess.jl")
include("src/quantizer.jl")
include("src/dataloader.jl")
include("src/model.jl")
include("src/loss.jl")

using .Data
using .Preprocess
using .Quantizer
using .DataLoader
using .Model
using .Loss

const USE_GPU = CUDA.functional()
to_device(x) = USE_GPU ? gpu(x) : x

function test_model_creation()
    model = Model.create_model()
    println("[Success] --> Model created with defaults")
    return model
end

function test_forward_pass(model)
    model = to_device(model)
    x = randn(Float32, 16000, 1, 2)
    x = to_device(x)
    y = model(x)
    println("[Success] --> Forward pass: $(size(x)) -> $(size(y))")
    return cpu(model)
end

function test_quantizer(model)
    model = to_device(model)
    x = randn(Float32, 16000, 1, 1)
    x = to_device(x)
    _, indices = Model.encode_quantize(model, x)
    println(
        "[Success] --> Quantizer: $(length(indices)) indices, range [$(minimum(indices)), $(maximum(indices))]",
    )
    return cpu(model)
end

function test_gradient_computation(model)
    model = to_device(model)
    x = randn(Float32, 16000, 1, 2)
    x = to_device(x)

    loss, grads = Flux.withgradient(model) do m
        y = m(x)
        Loss.composite_loss(y, x)
    end

    println("[Success] --> Gradients computed with composite loss = $loss")
    return cpu(model)
end

function test_dataloader()
    root = @__DIR__
    loader = DataLoader.MelDataLoader(root, "test-clean"; batch_size = 2)
    batch, _ = iterate(loader)
    println("[Success] --> DataLoader: batch shape = $(size(batch))")
end

function test_save_load(model)
    model_cpu = cpu(model)
    BSON.@save "test_model.bson" model_cpu=model_cpu
    data = BSON.load("test_model.bson")
    loaded = data[:model_cpu]
    x = randn(Float32, 16000, 1, 2)
    y = loaded(x)
    rm("test_model.bson")
    println("[Success] --> Save/load")
    return loaded
end

function run_all_tests()
    println("Testing ViT Neural Audio Coder...")
    model = test_model_creation()
    model = test_forward_pass(model)
    model = test_quantizer(model)
    model = test_gradient_computation(model)
    test_dataloader()
    model = test_save_load(model)
    println("\n[Success] --> All tests passed")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
