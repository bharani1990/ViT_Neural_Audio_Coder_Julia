module Preprocess

using ..Data

const PATCH_TIME = 16
const PATCH_FREQ = 16
const T_FIXED = 1024

function fix_length(mel::AbstractMatrix{<:Real}; T_fixed::Int = T_FIXED)
    n_mels, T = size(mel)
    if T == T_fixed
        return mel
    elseif T > T_fixed
        start = Int(floor((T - T_fixed) / 2)) + 1
        return mel[:, start:(start+T_fixed-1)]
    else
        padded = zeros(eltype(mel), n_mels, T_fixed)
        padded[:, 1:T] .= mel
        return padded
    end
end

function to_patches(
    mel_fixed::AbstractMatrix{<:Real};
    patch_freq::Int = PATCH_FREQ,
    patch_time::Int = PATCH_TIME,
)
    n_mels, T = size(mel_fixed)
    @assert n_mels % patch_freq == 0
    @assert T % patch_time == 0

    n_freq_patches = div(n_mels, patch_freq)
    n_time_patches = div(T, patch_time)
    num_patches = n_freq_patches * n_time_patches
    patch_dim = patch_freq * patch_time

    X = Array{Float32}(undef, num_patches, patch_dim)

    idx = 1
    for fy = 0:(n_freq_patches-1)
        for tx = 0:(n_time_patches-1)
            f_start = fy * patch_freq + 1
            t_start = tx * patch_time + 1
            patch =
                mel_fixed[f_start:(f_start+patch_freq-1), t_start:(t_start+patch_time-1)]
            X[idx, :] .= vec(Float32.(patch))
            idx += 1
        end
    end

    return X
end

end
