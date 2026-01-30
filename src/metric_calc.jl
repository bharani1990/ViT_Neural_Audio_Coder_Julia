module Metric_Calc

using ..Model
using ..Data
using ..Preprocess
using ..Quantizer
using Flux
using Statistics
using Printf
using JSON

const LATENCY_THRESHOLD = 20.0
const BITRATE_MIN = 8.0
const BITRATE_MAX = 16.0
const PESQ_THRESHOLD = 3.5
const STOI_THRESHOLD = 0.9

struct MetricResult
    latency::Float32
    bitrate::Float32
    pesq::Float32
    stoi::Float32
end

struct EvaluationReport
    metrics::Vector{MetricResult}
    sample_ids::Vector{String}
    avg_latency::Float32
    avg_bitrate::Float32
    avg_pesq::Float32
    avg_stoi::Float32
    latency_pass::Bool
    bitrate_pass::Bool
    pesq_pass::Bool
    stoi_pass::Bool
end

function estimate_pesq(original::AbstractArray, reconstructed::AbstractArray)
    mse = mean((original .- reconstructed) .^ 2)
    pesq = max(1.0f0, 4.5f0 - 0.5f0 * sqrt(mse))
    return min(4.5f0, pesq)
end

function estimate_stoi(original::AbstractArray, reconstructed::AbstractArray)
    eps = 1.0f-8
    orig_mean = mean(original)
    recon_mean = mean(reconstructed)
    orig_std = std(original)
    recon_std = std(reconstructed)
    
    orig_norm = (original .- orig_mean) ./ (orig_std + eps)
    recon_norm = (reconstructed .- recon_mean) ./ (recon_std + eps)
    correlation = mean(orig_norm .* recon_norm)
    stoi = (1.0f0 + correlation) / 2.0f0
    return clamp(stoi, 0.0f0, 1.0f0)
end

function evaluate_model(
    model,
    sample_audio::AbstractArray,
    sample_id::String
)
    target_length = 16000
    mel = Data.logmel(sample_audio)
    mel_fixed = Preprocess.fix_length_flat(mel, target_length)
    mel_input = reshape(mel_fixed, target_length, 1, 1)

    start_time = time()
    _, indices = Model.encode_quantize(model, mel_input)
    recon_output = model(mel_input)
    end_time = time()
    
    elapsed_time_ms = Float32((end_time - start_time) * 1000.0)
    latency = elapsed_time_ms + 5.0f0
    
    audio_duration_sec = Float32(length(sample_audio) / Data.SAMPLE_RATE)
    n_codes = model.quantizer.n_codes
    n_frames = length(indices)
    
    bits_per_frame = ceil(Int, log2(n_codes))
    total_bits = n_frames * bits_per_frame
    bitrate_kbps = Float32((total_bits / audio_duration_sec) / 1000.0)
    
    bitrate = max(bitrate_kbps, 0.1f0)
    
    recon_vec = vec(recon_output)
    orig_normalized = mel_fixed ./ (maximum(abs.(mel_fixed)) + 1.0f-8)
    recon_normalized = recon_vec ./ (maximum(abs.(recon_vec)) + 1.0f-8)
    
    min_len = min(length(orig_normalized), length(recon_normalized))
    pesq = estimate_pesq(orig_normalized[1:min_len], recon_normalized[1:min_len])
    stoi = estimate_stoi(orig_normalized[1:min_len], recon_normalized[1:min_len])

    return MetricResult(latency, bitrate, pesq, stoi)
end

function generate_report(metric_results::Vector{MetricResult}, sample_ids::Vector{String})
    if length(metric_results) <= 1
        avg_latency = metric_results[1].latency
        avg_bitrate = metric_results[1].bitrate
        avg_pesq = metric_results[1].pesq
        avg_stoi = metric_results[1].stoi
    else
        avg_latency = mean([m.latency for m in metric_results[2:end]])
        avg_bitrate = mean([m.bitrate for m in metric_results[2:end]])
        avg_pesq = mean([m.pesq for m in metric_results[2:end]])
        avg_stoi = mean([m.stoi for m in metric_results[2:end]])
    end
    
    latency_pass = avg_latency < LATENCY_THRESHOLD
    bitrate_pass = (avg_bitrate >= BITRATE_MIN) && (avg_bitrate <= BITRATE_MAX)
    pesq_pass = avg_pesq > PESQ_THRESHOLD
    stoi_pass = avg_stoi > STOI_THRESHOLD

    return EvaluationReport(
        metric_results,
        sample_ids,
        Float32(avg_latency),
        Float32(avg_bitrate),
        Float32(avg_pesq),
        Float32(avg_stoi),
        latency_pass,
        bitrate_pass,
        pesq_pass,
        stoi_pass
    )
end

function print_report(report::EvaluationReport)
    println("\n" * "="^70)
    println("NEURAL AUDIO CODER EVALUATION REPORT")
    println("="^70)

    println("\nSample-by-Sample Results:")
    println("-"^70)
    println("ID                           | Latency(ms) | Bitrate(kbps) | PESQ | STOI")
    println("-"^70)
    
    for (id, metric) in zip(report.sample_ids, report.metrics)
        @printf "%28s | %11.2f | %13.2f | %4.2f | %4.2f\n" id metric.latency metric.bitrate metric.pesq metric.stoi
    end

    println("\n" * "-"^70)
    println("AGGREGATED RESULTS (AVERAGE):")
    println("-"^70)
    @printf "Latency (ms):              %.2f ms (threshold: < %.1f ms) %s\n" report.avg_latency LATENCY_THRESHOLD (report.latency_pass ? "PASS" : "FAIL")
    @printf "Bitrate (kbps):            %.2f kbps (range: %.1f-%.1f kbps) %s\n" report.avg_bitrate BITRATE_MIN BITRATE_MAX (report.bitrate_pass ? "PASS" : "FAIL")
    @printf "PESQ Score (approx):       %.2f (threshold: > %.1f) %s\n" report.avg_pesq PESQ_THRESHOLD (report.pesq_pass ? "PASS" : "FAIL")
    @printf "STOI Score (approx):       %.4f (threshold: > %.2f) %s\n" report.avg_stoi STOI_THRESHOLD (report.stoi_pass ? "PASS" : "FAIL")
    println("="^70 * "\n")
end

function save_report(report::EvaluationReport, filepath::String)
    report_dict = Dict(
        "avg_metrics" => Dict(
            "latency_ms" => report.avg_latency,
            "bitrate_kbps" => report.avg_bitrate,
            "pesq" => report.avg_pesq,
            "stoi" => report.avg_stoi,
        ),
        "thresholds" => Dict(
            "latency_ms" => LATENCY_THRESHOLD,
            "bitrate_kbps" => [BITRATE_MIN, BITRATE_MAX],
            "pesq" => PESQ_THRESHOLD,
            "stoi" => STOI_THRESHOLD,
        ),
        "results" => Dict(
            "latency_pass" => report.latency_pass,
            "bitrate_pass" => report.bitrate_pass,
            "pesq_pass" => report.pesq_pass,
            "stoi_pass" => report.stoi_pass,
        ),
        "samples" => [
            Dict(
                "id" => id,
                "latency_ms" => m.latency,
                "bitrate_kbps" => m.bitrate,
                "pesq" => m.pesq,
                "stoi" => m.stoi,
            ) for (id, m) in zip(report.sample_ids, report.metrics)
        ],
    )

    open(filepath, "w") do f
        JSON.print(f, report_dict)
    end
    println("Report saved to $filepath")
end

end
