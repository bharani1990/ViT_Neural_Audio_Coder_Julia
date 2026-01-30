module Loss

using Statistics
using Flux

function spectral_convergence_loss(y_pred, y_true)
    eps = 1.0f-8
    numerator = sum((y_true .- y_pred) .^ 2)
    denominator = sum(y_true .^ 2) + eps
    return sqrt(numerator / denominator)
end

function log_magnitude_loss(y_pred, y_true)
    eps = 1.0f-8
    log_pred = log.(abs.(y_pred) .+ eps)
    log_true = log.(abs.(y_true) .+ eps)
    return mean(abs.(log_pred .- log_true))
end

function correlation_loss(y_pred, y_true)
    eps = 1.0f-8
    
    pred_mean = mean(y_pred)
    true_mean = mean(y_true)
    
    pred_centered = y_pred .- pred_mean
    true_centered = y_true .- true_mean
    
    numerator = sum(pred_centered .* true_centered)
    pred_var = sum(pred_centered .^ 2) + eps
    true_var = sum(true_centered .^ 2) + eps
    denominator = sqrt(pred_var * true_var)
    
    correlation = numerator / denominator
    return 1.0f0 - correlation
end

function composite_loss(y_pred, y_true, sr::Int=16000; 
                        w_mse::Float32=1.0f0, 
                        w_spectral::Float32=0.1f0, 
                        w_correlation::Float32=0.1f0)
    eps = 1.0f-8
    
    if any(isnan, y_pred) || any(isnan, y_true)
        return Float32(1.0f0)
    end
    
    if any(isinf, y_pred) || any(isinf, y_true)
        y_pred = clamp.(y_pred, -1.0f2, 1.0f2)
        y_true = clamp.(y_true, -1.0f2, 1.0f2)
    end
    
    mse = mean((y_pred .- y_true) .^ 2)
    
    if isnan(mse) || isinf(mse)
        mse = 1.0f0
    end
    
    mse = clamp(mse, 0.0f0, 10.0f0)
    
    spectral = spectral_convergence_loss(y_pred, y_true)
    corr = correlation_loss(y_pred, y_true)
    
    total_loss = w_mse * mse + w_spectral * spectral + w_correlation * corr
    
    if isnan(total_loss) || isinf(total_loss)
        return Float32(1.0f0)
    end
    
    total_loss = clamp(total_loss, 0.0f0, 100.0f0)
    
    return total_loss
end

export composite_loss

end
