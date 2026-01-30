using Flux
using CUDA
using LuxCUDA
using Random
using Printf
using JLD2

include("src/data.jl")
include("src/preprocess.jl")
include("src/quantizer.jl")
include("src/model.jl")
include("src/metric_calc.jl")

using .Data
using .Preprocess
using .Quantizer
using .Model
using .Metric_Calc

const DEFAULT_MODEL_PATH = "best_model.bson"
const DEFAULT_SPLIT = "test-clean"
const DEFAULT_NUM_SAMPLES = 5
const DEFAULT_REPORT_PATH = "evaluation_report.json"
const RNG_SEED = 20260117

function load_or_instantiate_model(model_path::AbstractString)
    if !isfile(model_path)
        println("Model file not found: $model_path")
        println("Creating new untrained model")
        return Model.create_model()
    end
    
    println("Loading trained model from: $model_path")
    model = JLD2.jldopen(model_path, "r") do file
        file["model"]
    end
    println("Model loaded successfully")
    return cpu(model)
end

function run_evaluation(;
    model_path::AbstractString = DEFAULT_MODEL_PATH,
    split::AbstractString = DEFAULT_SPLIT,
    num_samples::Int = DEFAULT_NUM_SAMPLES,
    report_path::AbstractString = DEFAULT_REPORT_PATH,
    rng_seed::Integer = RNG_SEED,
)
    root = @__DIR__
    rng = MersenneTwister(rng_seed)

    model = load_or_instantiate_model(model_path)
    model = cpu(model)

    files = Data.list_librispeech_flac(root; split = split)
    total = length(files)
    if total == 0
        error("No .flac files found.")
    end
    n = min(num_samples, total)
    permuted = Random.randperm(rng, total)
    selected_files = files[permuted[1:n]]

    metric_results = Vector{Metric_Calc.MetricResult}()
    sample_ids = Vector{String}()

    for file_path in selected_files
        id = relpath(file_path, root)
        @printf("Evaluating %s\n", id)
        sample_audio = Data.load_audio(file_path)
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
