module Model

using Flux
using CUDA
using NNlib
using ..Quantizer

struct SimpleMHA
    q_proj::Dense
    k_proj::Dense
    v_proj::Dense
    out_proj::Dense
    heads::Int
    dim::Int
end

function SimpleMHA(dim::Int, heads::Int)
    SimpleMHA(
        Dense(dim, dim),
        Dense(dim, dim),
        Dense(dim, dim),
        Dense(dim, dim),
        heads,
        dim,
    )
end

Flux.@layer SimpleMHA

function (mha::SimpleMHA)(x)
    B, T, D = size(x)
    head_dim = mha.dim ÷ mha.heads

    x_flat = reshape(permutedims(x, (3, 2, 1)), D, B * T)
    Q = mha.q_proj(x_flat)
    K = mha.k_proj(x_flat)
    V = mha.v_proj(x_flat)

    Q = permutedims(reshape(Q, mha.dim, T, B), (3, 2, 1))
    K = permutedims(reshape(K, mha.dim, T, B), (3, 2, 1))
    V = permutedims(reshape(V, mha.dim, T, B), (3, 2, 1))

    Q = reshape(Q, B * mha.heads, T, head_dim)
    K = reshape(K, B * mha.heads, T, head_dim)
    V = reshape(V, B * mha.heads, T, head_dim)

    scores = NNlib.batched_mul(Q, NNlib.batched_transpose(K)) ./ sqrt(Float32(head_dim))
    attn = softmax(scores, dims = 3)
    out = NNlib.batched_mul(attn, V)

    out = reshape(out, B, mha.heads, T, head_dim)
    out = permutedims(out, (1, 3, 2, 4))
    out = reshape(out, B, T, mha.dim)

    out_flat = reshape(permutedims(out, (3, 2, 1)), mha.dim, B * T)
    out_proj = mha.out_proj(out_flat)
    out = permutedims(reshape(out_proj, mha.dim, T, B), (3, 2, 1))

    return out
end

struct TransformerBlock
    attention::SimpleMHA
    norm1::LayerNorm
    norm2::LayerNorm
    mlp::Chain
    dropout::Dropout
end

function TransformerBlock(dim::Int, heads::Int, mlp_dim::Int; dropout_rate = 0.1)
    TransformerBlock(
        SimpleMHA(dim, heads),
        LayerNorm(dim),
        LayerNorm(dim),
        Chain(
            Dense(dim, mlp_dim, gelu),
            Dropout(dropout_rate),
            Dense(mlp_dim, dim),
            Dropout(dropout_rate),
        ),
        Dropout(dropout_rate),
    )
end

Flux.@layer TransformerBlock

function (block::TransformerBlock)(x)
    B, T, D = size(x)

    x_flat = reshape(permutedims(x, (3, 2, 1)), D, B * T)
    normed1 = block.norm1(x_flat)
    normed1 = permutedims(reshape(normed1, D, T, B), (3, 2, 1))

    attn_out = block.attention(normed1)
    x = x .+ block.dropout(attn_out)

    x_flat = reshape(permutedims(x, (3, 2, 1)), D, B * T)
    normed2 = block.norm2(x_flat)
    normed2 = permutedims(reshape(normed2, D, T, B), (3, 2, 1))

    normed2_flat = reshape(permutedims(normed2, (3, 2, 1)), D, B * T)
    mlp_out = block.mlp(normed2_flat)
    mlp_out = permutedims(reshape(mlp_out, D, T, B), (3, 2, 1))

    x = x .+ mlp_out
    return x
end

struct PatchEmbed
    proj::Conv
    norm::LayerNorm
end

function PatchEmbed(in_chans::Int, embed_dim::Int, patch_size::Int)
    PatchEmbed(
        Conv((patch_size,), in_chans => embed_dim, stride = patch_size),
        LayerNorm(embed_dim),
    )
end

Flux.@layer PatchEmbed

function (pe::PatchEmbed)(x)
    x = pe.proj(x)
    B = size(x, 3)
    T = size(x, 1)
    D = size(x, 2)
    x = reshape(x, T, D, B)
    x = permutedims(x, (3, 1, 2))

    x_flat = reshape(permutedims(x, (3, 2, 1)), D, B * T)
    x_normed = pe.norm(x_flat)
    x = permutedims(reshape(x_normed, D, T, B), (3, 2, 1))

    return x
end

struct AudioEncoder
    patch_embed::PatchEmbed
    pos_embed::AbstractArray
    blocks::Vector{TransformerBlock}
    norm::LayerNorm
    dropout::Dropout
end

function AudioEncoder(
    in_chans::Int,
    embed_dim::Int,
    depth::Int,
    heads::Int,
    mlp_dim::Int,
    patch_size::Int,
    max_len::Int;
    dropout_rate = 0.1,
)
    num_patches = max_len ÷ patch_size

    AudioEncoder(
        PatchEmbed(in_chans, embed_dim, patch_size),
        randn(Float32, 1, num_patches, embed_dim) .* 0.02f0,
        [TransformerBlock(embed_dim, heads, mlp_dim; dropout_rate) for _ = 1:depth],
        LayerNorm(embed_dim),
        Dropout(dropout_rate),
    )
end

Flux.@layer AudioEncoder

function (enc::AudioEncoder)(x)
    x = enc.patch_embed(x)
    x = x .+ enc.pos_embed
    x = enc.dropout(x)

    for block in enc.blocks
        x = block(x)
    end

    B, T, D = size(x)
    x_flat = reshape(permutedims(x, (3, 2, 1)), D, B * T)
    x_normed = enc.norm(x_flat)
    x = permutedims(reshape(x_normed, D, T, B), (3, 2, 1))

    return x
end

struct AudioDecoder
    blocks::Vector{TransformerBlock}
    norm::LayerNorm
    head::Chain
    unpatch::ConvTranspose
end

Flux.@layer AudioDecoder

function AudioDecoder(
    embed_dim::Int,
    depth::Int,
    heads::Int,
    mlp_dim::Int,
    out_chans::Int,
    patch_size::Int;
    dropout_rate = 0.1,
    enc_depth::Int = 4,
)
    AudioDecoder(
        [TransformerBlock(embed_dim, heads, mlp_dim; dropout_rate) for _ = 1:depth],
        LayerNorm(embed_dim),
        Chain(Dense(embed_dim, embed_dim, gelu), Dense(embed_dim, embed_dim)),
        ConvTranspose((patch_size,), embed_dim => out_chans, stride = patch_size),
    )
end

Flux.@layer AudioDecoder

function (dec::AudioDecoder)(x)
    for block in dec.blocks
        x = block(x)
    end

    B, T, D = size(x)
    x_flat = reshape(permutedims(x, (3, 2, 1)), D, B * T)
    x_normed = dec.norm(x_flat)
    x = permutedims(reshape(x_normed, D, T, B), (3, 2, 1))

    x_flat = reshape(permutedims(x, (3, 2, 1)), D, B * T)
    x_head = dec.head(x_flat)
    x = permutedims(reshape(x_head, D, T, B), (3, 2, 1))

    x = permutedims(x, (2, 3, 1))
    x = dec.unpatch(x)

    return x
end

struct ViTAudioCoder
    encoder::AudioEncoder
    quantizer::Quantizer.VectorQuantizer
    decoder::AudioDecoder
end

function ViTAudioCoder(;
    in_chans = 1,
    out_chans = 1,
    embed_dim = 96,
    enc_depth = 2,
    dec_depth = 1,
    heads = 4,
    mlp_dim = 192,
    patch_size = 160,
    max_len = 16000,
    dropout_rate = 0.1,
    n_codes = 256,
    target_bitrate_kbps = 12.0f0,
)
    ViTAudioCoder(
        AudioEncoder(
            in_chans,
            embed_dim,
            enc_depth,
            heads,
            mlp_dim,
            patch_size,
            max_len;
            dropout_rate,
        ),
        Quantizer.VectorQuantizer(n_codes, embed_dim; target_bitrate_kbps=target_bitrate_kbps),
        AudioDecoder(
            embed_dim,
            dec_depth,
            heads,
            mlp_dim,
            out_chans,
            patch_size;
            dropout_rate,
        ),
    )
end

Flux.@layer ViTAudioCoder

function (model::ViTAudioCoder)(x)
    latent = model.encoder(x)
    quantized, _ = model.quantizer(latent)
    output = model.decoder(quantized)
    return output
end

function encode_quantize(model::ViTAudioCoder, x)
    latent = model.encoder(x)
    quantized, indices = model.quantizer(latent)
    indices_cpu = Array(indices)
    return quantized, indices_cpu
end

function create_model(; kwargs...)
    model = ViTAudioCoder(; kwargs...)
    return model
end

function compute_latency_ms(sample_rate::Int, patch_size::Int)
    frame_samples = patch_size
    latency_sec = Float32(frame_samples) / Float32(sample_rate)
    latency_ms = latency_sec * 1000.0f0
    return latency_ms
end

export ViTAudioCoder, create_model, AudioEncoder, AudioDecoder, encode_quantize, compute_latency_ms

end
