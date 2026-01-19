using Flux
using CUDA
using LuxCUDA
using BSON
using Random
using Printf

include("src/data.jl")
include("src/preprocess.jl")
include("src/quantizer.jl")
include("src/dataset.jl")
include("src/dataloader.jl")
include("src/model.jl")
include("src/metric_calc.jl")

using .Data
using .Preprocess
using .Quantizer
using .Dataset
using .Model
using .Metric_Calc

const DEFAULT_MODEL_PATH = "best_model.bson"
const DEFAULT_SPLIT = "test-clean"
const DEFAULT_NUM_SAMPLES = 5
const DEFAULT_REPORT_PATH = "evaluation_report.json"
const RNG_SEED = 20260117

function resolve_model_checkpoint(model_path::AbstractString)
    if !isfile(model_path)
        return nothing
    end
    data = BSON.load(model_path)
    for key in (:model, :model_cpu)
        if haskey(data, key)
            return data[key]
        end
    end
    for (key, value) in data
        if occursin("model", lowercase(string(key)))
            return value
        end
    end
    return nothing
end

function build_model()
    return Model.create_model(
        in_chans=1,
        out_chans=1,
        embed_dim=256,
        enc_depth=6,
        dec_depth=4,
        heads=8,
        mlp_dim=1024,
        patch_size=16,
        max_len=16384,
        dropout_rate=0.1
    )
end

function select_audio_files(files::Vector{String}, n::Int, rng::AbstractRNG)
    total = length(files)
    if total == 0
        error("No .flac files found.")
    end
    n = min(n, total)
    permuted = Random.randperm(rng, total)
    return files[permuted[1:n]]
end

function sample_id(path::AbstractString, root::AbstractString)
    return relpath(path, root)
end

function prepare_audio_sample(path::AbstractString)
    audio = Data.load_audio(path)
    return audio
end

function load_or_instantiate_model(model_path::AbstractString)
    if !isfile(model_path)
        error("Model file not found: $model_path")
    end
    
    checkpoint_model = resolve_model_checkpoint(model_path)
    if checkpoint_model !== nothing
        @printf("Loaded model from %s\n", model_path)
        return cpu(checkpoint_model)
    end
    
    error("Failed to load model from $model_path - no valid model found in checkpoint")
end

function run_evaluation(; model_path::AbstractString=DEFAULT_MODEL_PATH,
                        split::AbstractString=DEFAULT_SPLIT,
                        num_samples::Int=DEFAULT_NUM_SAMPLES,
                        report_path::AbstractString=DEFAULT_REPORT_PATH,
                        rng_seed::Integer=RNG_SEED)
    root = @__DIR__
    rng = MersenneTwister(rng_seed)

    model = load_or_instantiate_model(model_path)
    model = cpu(model)

    files = Data.list_librispeech_flac(root; split=split)
    selected_files = select_audio_files(files, num_samples, rng)

    metric_results = Vector{Metric_Calc.MetricResult}()
    sample_ids = Vector{String}()

    for file_path in selected_files
        id = sample_id(file_path, root)
        @printf("Evaluating %s\n", id)
        sample_audio = prepare_audio_sample(file_path)
        push!(sample_ids, id)
        push!(metric_results, Metric_Calc.evaluate_model(model, sample_audio, id))
    end

    report = Metric_Calc.generate_report(metric_results, sample_ids)
    Metric_Calc.print_report(report)
    Metric_Calc.save_report(report, report_path)
end

function main()
    run_evaluation()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
