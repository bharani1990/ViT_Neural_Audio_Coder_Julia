module Dataset

using ..Data
using ..Preprocess
using Random

struct AudioDataset
    files::Vector{String}
end

function AudioDataset(root::AbstractString; split::String = "train-clean")
    files = Data.list_librispeech_flac(root; split = split)
    return AudioDataset(files)
end

Base.length(ds::AudioDataset) = length(ds.files)

function get_example(ds::AudioDataset, i::Int)
    path = ds.files[i]
    mel = Data.load_logmel(path)
    mel_fixed = Preprocess.fix_length(mel)
    patches = Preprocess.to_patches(mel_fixed)
    return patches
end

function shuffled_indices(ds::AudioDataset; rng = Random.GLOBAL_RNG)
    return Random.shuffle(rng, collect(1:length(ds)))
end

end
