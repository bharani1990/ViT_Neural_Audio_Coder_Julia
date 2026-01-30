# ViT Neural Audio Coder (Julia)

A Vision Transformer (ViT)-based neural audio codec implemented in Julia using Flux.jl. This project encodes and decodes audio using transformer architectures with vector quantization.

## Project Overview

This neural audio coder uses:
- **Vision Transformer (ViT)** architecture adapted for audio
- **Vector Quantization (VQ)** for discrete latent codes
- **Encoder-Decoder** structure for audio reconstruction
- **LibriSpeech dataset** for training and evaluation

## To Reproduce at your end!

### Prerequisites - very important - otherwise, run at your own risk :) 

- Julia 1.9 or higher (Why Julia - But, Why not Julia?)
- CUDA-capable GPU (optional, but recommended for training) (Run in your CPU as well - but go for a vacation and come back!)

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/ViT_Neural_Audio_Coder_Julia
   ```

2. **Activate the Julia environment and install dependencies**
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

   I made it simpler for you! Thank me later :) 

   This will install all required packages listed in `Project.toml`: 
   - Flux, CUDA, LuxCUDA (deep learning)
   - WAV, LibSndFile, SampledSignals (audio processing)
   - FFTW (spectral analysis)
   - Plots, Images (visualization)
   - And more...

## How to use the scripts?

### 1. Test Model Architecture

Before training, verify that the model is correctly constructed: (Trust me, run this to see some nice Bland logs)

```bash
julia --project=. test_model.jl
```

### 2. Train From Scratch

To train a new model from the beginning: (for the first time, you definitely have to! Otherwise, i have the .bson as well to continue training from 30 epochs!). This will save the model to `best_model.bson` while display training progress and loss

```bash
julia --project=. main.jl 
```

**Configuration**: Edit hyperparameters in `main.jl`:
- `embed_dim`: Embedding dimension (default: 256)
- `enc_depth`: Encoder depth (default: 6)
- `dec_depth`: Decoder depth (default: 4)
- `n_codes`: Codebook size (default: 1024)
- `num_epochs`: Training epochs (default: 30)
- `batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 1e-4)

### 3. Train From Checkpoint

To resume training from a saved checkpoint:

```bash
julia main.jl --continue best_model.bson
```

The script automatically detects and loads `best_model.bson` if available. The training process always saves the best-performing model to `best_model.bson`, which is referenced throughout the codebase.

### 4. Evaluate Model

Evaluate a trained model on test data: Sadly, i will agree that this is not a great encoder - decoder architecture, but i tried! The PESQ and STOI is relatively ok, but bit-rate and latency - I am afraid, Not proud numbers to boast off! But, that gives a chance for you to improve. Go for it. 

```bash
julia --project=. evaluate.jl
```

**Options** (edit in `evaluate.jl`):
- `DEFAULT_MODEL_PATH`: Path to model file (default: `"best_model.bson"`)
- `DEFAULT_SPLIT`: Dataset split (default: `"test-clean"`)
- `DEFAULT_NUM_SAMPLES`: Number of samples to evaluate (default: 5)

### 5. Run Audio Inspection Notebook

Visualize and analyze audio processing by opening `audio_inspection.ipynb` in VS Code or Jupyter and running all cells after activating the Julia Kernel (ofcourse - how else can you run the cell in a notebook!)

The notebook includes:
- Load and play audio samples
- Visualize waveforms and spectrograms
- Inspect model predictions
- Compare original vs. reconstructed audio
- Analyze frequency content
- Test data preprocessing pipeline

## Project Structure

```
ViT_Neural_Audio_Coder_Julia/
├── data/
│   └── LibriSpeech/             # Dataset (train/dev/test splits)
├── src/
│   ├── checkpoint.jl           # Checkpoint management
│   ├── data.jl                 # Data loading utilities
│   ├── dataloader.jl           # Data batching
│   ├── dataset.jl              # Dataset definitions
│   ├── device.jl               # GPU/CPU device handling
|   |── metric_calc.jl          # Evaluation metrics
|   |── model.jl                # ViT model architecture
|   |── preprocess.jl           # Audio preprocessing 
│   ├── quantizer.jl            # Vector quantization
│   ├── train.jl                # Training utilities
├──.gitignore                   # You know why!
├── main.jl                      # Main training script
├── test_model.jl                # Model architecture tests
├── evaluate.jl                  # Model evaluation script
├── audio_inspection.ipynb       # Visualization notebook
├── Project.toml                 # Julia dependencies
├── Manifest.toml                # Dependency lock file
├── best_model.bson              # Best trained model (after training)
├── evaluation_report.json       # Evaluation metrics (after evaluation)
├── input_audio.wav              # Indicative Name, yeah!
├── reconstructed_audio.wav      # Again, Indicative Name, yeah!
├── losses.json                  # Saving it to plot the loss plot!
├── README.md                    # Its me, Recursively calling me
```

## Key Features

- **GPU Acceleration**: Automatic GPU detection and utilization
- **Checkpoint System**: Resume training from any epoch
- **Comprehensive Metrics**: MSE, SNR, PESQ for audio quality
- **Progress Tracking**: Real-time training progress bars
- **Modular Design**: Clean separation of concerns across modules

## Dataset

This project uses the **LibriSpeech** dataset:
- `train-clean`: Training data
- `dev-clean`: Validation data
- `test-clean`: Test data

Place your LibriSpeech dataset in `data/LibriSpeech/`.

## Troubleshooting

**CUDA issues**: If GPU is not detected:
```julia
# In Julia REPL
using CUDA
CUDA.functional()
```

**Dependency conflicts**: Reinstall dependencies:
```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

**Out of memory**: Reduce batch size in `main.jl`:
```julia
batch_size = 2 # I am very poor, so i could afford only this, but increase to your convenient number. (exponent of 2 is preferable)
```

## Citation

If you use this code in your research, please cite:
```
@software{vit_neural_audio_coder_julia,
  author = {Bharani}
  title = {ViT Neural Audio Coder (Julia Implementation)},
  year = {2026}
}
```

## License

See individual files for licensing information. LibriSpeech dataset has its own license terms.

## Contributing

Contributions are welcome! If you care, Otherwise, its ok...wait, I am joking. This is the only area where i joked honestly. So, on a serious note, kindly do whatever deems good to you. Thank you! 