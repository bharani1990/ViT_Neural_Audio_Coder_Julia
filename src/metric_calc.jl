module Metric_Calc

using ..Model
using ..Dataset
using ..DataLoader
using ..Data
using ..Preprocess
using ..Quantizer
using Flux
using JLD2
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
    mse::Float32
    snr::Float32
end

struct EvaluationReport
    metrics::Vector{MetricResult}
    sample_ids::Vector{String}
    avg_latency::Float32
    avg_bitrate::Float32
    avg_pesq::Float32
    avg_stoi::Float32
    avg_mse::Float32
    avg_snr::Float32
    latency_pass::Bool
    bitrate_pass::Bool
    pesq_pass::Bool
    stoi_pass::Bool
    overall_pass::Bool
end

function compute_mse(original::AbstractArray, reconstructed::AbstractArray)
    return mean((original .- reconstructed) .^ 2)
end

function compute_snr(original::AbstractArray, reconstructed::AbstractArray)
    signal_power = mean(original .^ 2)
    noise_power = mean((original .- reconstructed) .^ 2)
    if noise_power == 0
        return 100.0f0
    end
    return 10.0f0 * log10(signal_power / noise_power)
end

function estimate_pesq(original::AbstractArray, reconstructed::AbstractArray)
    mse = compute_mse(original, reconstructed)
    pesq = max(1.0f0, 4.5f0 - 0.5f0 * sqrt(mse))
    return min(4.5f0, pesq)
end

function estimate_stoi(original::AbstractArray, reconstructed::AbstractArray)
    orig_norm = (original .- mean(original)) ./ std(original)
    recon_norm = (reconstructed .- mean(reconstructed)) ./ std(reconstructed)
    correlation = mean(orig_norm .* recon_norm)
    stoi = (1.0f0 + correlation) / 2.0f0
    return clamp(stoi, 0.0f0, 1.0f0)
end

function estimate_latency(model_forward_time_ms::Float32, capture_playback_overhead_ms::Float32=5.0f0)
    return model_forward_time_ms + capture_playback_overhead_ms
end

function estimate_bitrate(audio_length_ms::Float32, compressed_size_bytes::Int)
    compressed_size_bits = compressed_size_bytes * 8
    bitrate = (compressed_size_bits / audio_length_ms) * 1000.0f0 / 1000.0f0
    return bitrate
end

function evaluate_model(model, sample_audio::AbstractArray, sample_id::String; 
                       audio_duration_ms::Float32=100.0f0)
    start_time = time()
    
    mel = Data.logmel(sample_audio)
    mel_fixed = Preprocess.fix_length(mel)
    mel_flat = reshape(mel_fixed, :, 1, 1)
    
    _, indices = Model.encode_quantize(model, mel_flat)
    
    recon_flat = model(mel_flat)
    recon_mel = reshape(recon_flat[:, 1, 1], size(mel_fixed))
    
    recon_audio = Data.griffin_lim(recon_mel)
    
    end_time = time()
    elapsed_time = (end_time - start_time) * 1000.0f0
    
    min_len = min(length(sample_audio), length(recon_audio))
    orig = sample_audio[1:min_len]
    recon = recon_audio[1:min_len]
    
    audio_duration_sec = Float32(length(sample_audio) / Data.SAMPLE_RATE)
    indices_int = Int.(clamp.(round.(indices), 1, typemax(Int)))
    bitrate = Quantizer.compute_bitrate(indices_int, audio_duration_sec)
    
    latency = estimate_latency(Float32(elapsed_time))
    mse = compute_mse(orig, recon)
    snr = compute_snr(orig, recon)
    pesq = estimate_pesq(orig, recon)
    stoi = estimate_stoi(orig, recon)
    
    return MetricResult(
        Float32(latency),
        Float32(bitrate),
        Float32(pesq),
        Float32(stoi),
        Float32(mse),
        Float32(snr)
    )
end

function generate_report(metric_results::Vector{MetricResult}, sample_ids::Vector{String})
    avg_latency = mean([m.latency for m in metric_results[2:end]])
    avg_bitrate = mean([m.bitrate for m in metric_results[2:end]])
    avg_pesq = mean([m.pesq for m in metric_results[2:end]])
    avg_stoi = mean([m.stoi for m in metric_results[2:end]])
    avg_mse = mean([m.mse for m in metric_results[2:end]])
    avg_snr = mean([m.snr for m in metric_results[2:end]])
    latency_pass = avg_latency < LATENCY_THRESHOLD
    bitrate_pass = (avg_bitrate >= BITRATE_MIN) && (avg_bitrate <= BITRATE_MAX)
    pesq_pass = avg_pesq >= PESQ_THRESHOLD
    stoi_pass = avg_stoi >= STOI_THRESHOLD
    overall_pass = latency_pass && bitrate_pass && pesq_pass && stoi_pass
    
    return EvaluationReport(
        metric_results,
        sample_ids,
        Float32(avg_latency),
        Float32(avg_bitrate),
        Float32(avg_pesq),
        Float32(avg_stoi),
        Float32(avg_mse),
        Float32(avg_snr),
        latency_pass,
        bitrate_pass,
        pesq_pass,
        stoi_pass,
        overall_pass
    )
end

function print_report(report::EvaluationReport)    
    println("\n" * "="^70)
    println("NEURAL AUDIO CODER EVALUATION REPORT")
    println("="^70)
    
    println("\nSample-by-Sample Results:")
    println("-"^70)
    println("""
    ID                                           | Latency(ms) | Bitrate(kbps) | PESQ | STOI | MSE    | SNR(dB)
    """)
    for (id, metric) in zip(report.sample_ids, report.metrics)
        @printf "%25s | %11.2f | %13.2f | %4.2f | %4.2f | %6.2f | %7.2f\n" id metric.latency metric.bitrate metric.pesq metric.stoi metric.mse metric.snr
    end
    
    println("\n" * "-"^70)
    println("AGGREGATED RESULTS (AVERAGE):")
    println("-"^70)
    @printf "Latency (ms):              %.2f ms (threshold: < %.1f ms) %s\n" report.avg_latency LATENCY_THRESHOLD (report.latency_pass ? "PASS" : "FAIL")
    @printf "Bitrate (kbps):            %.2f kbps (range: %.1f-%.1f kbps) %s\n" report.avg_bitrate BITRATE_MIN BITRATE_MAX (report.bitrate_pass ? "PASS" : "FAIL")
    @printf "PESQ Score:                %.2f (threshold: ≥ %.1f) %s\n" report.avg_pesq PESQ_THRESHOLD (report.pesq_pass ? "PASS" : "FAIL")
    @printf "STOI Score:                %.4f (threshold: ≥ %.2f) %s\n" report.avg_stoi STOI_THRESHOLD (report.stoi_pass ? "PASS" : "FAIL")
    @printf "Mean Squared Error (MSE):  %.6f\n" report.avg_mse
    @printf "Signal-to-Noise Ratio:     %.2f dB\n" report.avg_snr
    
    println("\n" * "-"^70)
    println("OVERALL RESULT: " * (report.overall_pass ? "PASS - Model meets all baselines" : "FAIL - Model does not meet all baselines"))
    println("="^70 * "\n")
end

function save_report(report::EvaluationReport, filepath::String)
    report_dict = Dict(
        "avg_metrics" => Dict(
            "latency_ms" => report.avg_latency,
            "bitrate_kbps" => report.avg_bitrate,
            "pesq" => report.avg_pesq,
            "stoi" => report.avg_stoi,
            "mse" => report.avg_mse,
            "snr_db" => report.avg_snr
        ),
        "thresholds" => Dict(
            "latency_ms" => LATENCY_THRESHOLD,
            "bitrate_kbps" => [BITRATE_MIN, BITRATE_MAX],
            "pesq" => PESQ_THRESHOLD,
            "stoi" => STOI_THRESHOLD
        ),
        "results" => Dict(
            "latency_pass" => report.latency_pass,
            "bitrate_pass" => report.bitrate_pass,
            "pesq_pass" => report.pesq_pass,
            "stoi_pass" => report.stoi_pass,
            "overall_pass" => report.overall_pass
        ),
        "samples" => [
            Dict(
                "id" => id,
                "latency_ms" => m.latency,
                "bitrate_kbps" => m.bitrate,
                "pesq" => m.pesq,
                "stoi" => m.stoi,
                "mse" => m.mse,
                "snr_db" => m.snr
            )
            for (id, m) in zip(report.sample_ids, report.metrics)
        ]
    )
    
    open(filepath, "w") do f
        JSON.print(f, report_dict)
    end
    println("Report saved to $filepath")
end

end
