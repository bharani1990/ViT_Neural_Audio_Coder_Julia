module DataLoader

using ..Data
using ..Preprocess
using Random

struct MelDataLoader
    files::Vector{String}
    batch_size::Int
    indices::Vector{Int}
    target_length::Int
end

function MelDataLoader(
    root::String,
    split::String;
    batch_size::Int = 4,
    target_length::Int = 16000,
    rng = Random.GLOBAL_RNG,
)
    files = Data.list_librispeech_flac(root; split = split)
    indices = Random.shuffle(rng, collect(1:length(files)))
    return MelDataLoader(files, batch_size, indices, target_length)
end

Base.IteratorSize(::Type{MelDataLoader}) = Base.HasLength()
Base.length(loader::MelDataLoader) = div(length(loader.indices), loader.batch_size)

function Base.iterate(loader::MelDataLoader, state::Int = 1)
    n = length(loader.indices)
    state > n && return nothing

    batch_end = min(state + loader.batch_size - 1, n)
    batch_idx = loader.indices[state:batch_end]

    mel_list = []
    for idx in batch_idx
        try
            mel = Data.load_logmel(loader.files[idx])
            mel_fixed = Preprocess.fix_length_flat(mel, loader.target_length)
            push!(mel_list, mel_fixed)
        catch
            continue
        end
    end

    isempty(mel_list) && return Base.iterate(loader, state + loader.batch_size)

    T = length(mel_list[1])
    b = length(mel_list)
    batch = Array{Float32}(undef, T, 1, b)

    for (j, mel_vec) in enumerate(mel_list)
        batch[:, 1, j] .= mel_vec
    end

    return batch, state + loader.batch_size
end

end
