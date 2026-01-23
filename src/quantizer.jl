module Quantizer

using Flux
using Statistics
using Functors

struct VectorQuantizer
    codebook::AbstractArray{Float32}
    n_codes::Int
    dim::Int
end

function VectorQuantizer(n_codes::Int, dim::Int)
    cb = randn(Float32, dim, n_codes) .* 0.02f0
    VectorQuantizer(cb, n_codes, dim)
end

Flux.@layer VectorQuantizer
Functors.@functor VectorQuantizer

function (vq::VectorQuantizer)(z)
    B, T, D = size(z)

    zf = reshape(permutedims(z, (3, 2, 1)), D, B * T)

    z_squared = sum(zf .* zf, dims = 1)
    cb_squared = sum(vq.codebook .* vq.codebook, dims = 1)

    distances = z_squared' .+ cb_squared .- 2.0f0 .* (zf' * vq.codebook)

    _, min_idx_2d = findmin(distances, dims = 2)

    indices_range = reshape(1:vq.n_codes, 1, vq.n_codes)
    one_hot = Float32.(reshape(min_idx_2d, :, 1) .== indices_range)

    quantized_flat = vq.codebook * one_hot'
    quantized_reshaped = permutedims(reshape(quantized_flat, D, T, B), (3, 2, 1))

    straight_through = z .+ (quantized_reshaped .- z)

    indices_for_bitrate = dropdims(sum(one_hot .* indices_range, dims = 2), dims = 2)

    return straight_through, indices_for_bitrate
end

function compute_bitrate(indices::Vector{Int}, audio_duration_sec::Float32)
    n_codes_transmitted = length(indices)
    bits_per_code = ceil(Int, log2(maximum(indices)))
    total_bits = n_codes_transmitted * bits_per_code
    bitrate_bps = total_bits / audio_duration_sec
    bitrate_kbps = bitrate_bps / 1000.0f0
    return Float32(bitrate_kbps)
end

export VectorQuantizer, compute_bitrate

end
