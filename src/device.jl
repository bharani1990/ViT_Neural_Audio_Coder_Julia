module Device

using CUDA
using LuxCUDA
using Flux

export get_device, setup_device, to_device

function setup_device()
    if CUDA.has_cuda()
        @info "CUDA available: $(CUDA.devices())"
        device = gpu
        backend = "GPU"
    else
        @info "CUDA not available, using CPU"
        device = cpu
        backend = "CPU"
    end
    return device, backend
end

function get_device()::Function
    CUDA.has_cuda() ? gpu : cpu
end

function to_device(x, device::Function)
    device(x)
end

function to_device(x::Nothing, device::Function)
    nothing
end

end
