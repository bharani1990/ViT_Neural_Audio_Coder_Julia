module Data

using FileIO
using LibSndFile
using FFTW
using LinearAlgebra
using Statistics

const SAMPLE_RATE = 16_000
const N_FFT       = 512
const HOP_LENGTH  = 160
const N_MELS      = 80
const FMIN        = 50.0
const FMAX        = 8000.0


const HANN_WIN = 0.5 .- 0.5 .* cos.(2π .* (0:N_FFT-1) ./ (N_FFT-1))

function stft_mag(x::Vector{Float32};
                  n_fft::Int=N_FFT,
                  hop_length::Int=HOP_LENGTH)
    xpad = vcat(zeros(Float32, div(n_fft,2)), x, zeros(Float32, div(n_fft,2)))
    n_frames = cld(length(xpad) - n_fft, hop_length) + 1
    S = Matrix{Float32}(undef, div(n_fft,2)+1, n_frames)

    for i in 1:n_frames
        start = (i-1)*hop_length + 1
        stop  = start + n_fft - 1
        if stop > length(xpad)
            frame = zeros(Float32, n_fft)
            len = max(0, length(xpad) - start + 1)
            len > 0 && copyto!(frame, 1, xpad, start, len)
        else
            frame = @view xpad[start:stop]
        end
        windowed = HANN_WIN .* frame
        spec = rfft(windowed)
        S[:, i] = abs.(spec)
    end

    return S
end

mel(f) = 2595.0 * log10(1 + f / 700.0)
mel_inv(m) = 700.0 * (10.0^(m / 2595.0) - 1.0)

function mel_filterbank(sr::Int=SAMPLE_RATE;
                        n_fft::Int=N_FFT,
                        n_mels::Int=N_MELS,
                        fmin::Float64=FMIN,
                        fmax::Float64=FMAX)
    n_freqs = div(n_fft,2) + 1
    fmin_mel = mel(fmin)
    fmax_mel = mel(fmax)
    mels = range(fmin_mel, fmax_mel, length=n_mels+2)
    hz = mel_inv.(mels)

    bins = floor.(Int, (n_fft .+ 1) .* hz ./ (2 * sr))
    fb = zeros(Float32, n_mels, n_freqs)

    for m in 1:n_mels
        f_m_left   = bins[m]
        f_m_center = bins[m+1]
        f_m_right  = bins[m+2]

        for k in f_m_left:f_m_center
            if 1 ≤ k ≤ n_freqs
                denom = max(f_m_center - f_m_left, 1)
                fb[m, k] = (k - f_m_left) / denom
            end
        end

        for k in f_m_center:f_m_right
            if 1 ≤ k ≤ n_freqs
                denom = max(f_m_right - f_m_center, 1)
                fb[m, k] = (f_m_right - k) / denom
            end
        end
    end

    return fb
end

const MEL_FB = mel_filterbank()

function logmel(x::Vector{Float32})
    S = stft_mag(x)
    M = MEL_FB * S
    M .= max.(M, 1.0f-10)
    return log.(M)
end

function load_audio(path::AbstractString)
    buf = FileIO.load(path)

    sr = try
        Int(round(buf.samplerate))
    catch
        SAMPLE_RATE
    end

    sr != SAMPLE_RATE && error("Expected sr = $(SAMPLE_RATE), but got $sr for $path")

    a = Array(buf)
    x_vec = ndims(a) == 1 ? Float32.(a) : Float32.(a[:, 1])

    return x_vec
end

function load_logmel(path::AbstractString)
    x = load_audio(path)
    return logmel(x)
end

function griffin_lim(logmel_spec::Matrix{Float32}; n_iter::Int=32)
    S = exp.(logmel_spec)
    S_linear = MEL_FB' * S
    S_linear = max.(S_linear, 1.0f-10)
    
    phase = 2π .* rand(Float32, size(S_linear))
    for _ in 1:n_iter
        S_complex = S_linear .* exp.(im .* phase)
        x = irfft(S_complex, N_FFT)
        S_reconstructed = rfft(HANN_WIN .* x)
        phase = angle.(S_reconstructed)
    end
    
    S_complex = S_linear .* exp.(im .* phase)
    frames = [irfft(S_complex[:, i], N_FFT) for i in 1:size(S_complex, 2)]
    
    output_len = HOP_LENGTH * (length(frames) - 1) + N_FFT
    audio = zeros(Float32, output_len)
    for (i, frame) in enumerate(frames)
        start = (i-1) * HOP_LENGTH + 1
        audio[start:start+N_FFT-1] .+= frame .* HANN_WIN
    end
    
    return audio[div(N_FFT,2)+1:end-div(N_FFT,2)]
end

function list_librispeech_flac(root::AbstractString;
                               split::String = "train-clean")
    base = joinpath(root, "data", "LibriSpeech", split)
    files = String[]
    for (dirpath, _, fnames) in walkdir(base)
        for f in fnames
            endswith(f, ".flac") && push!(files, joinpath(dirpath, f))
        end
    end
    sort!(files)
    return files
end

end
