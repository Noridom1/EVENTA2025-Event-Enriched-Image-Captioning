# Local Gemma Image Captioning

This directory contains a local implementation of Google's quantized Gemma 3 27B model for image captioning, with comprehensive logging and monitoring capabilities.

## 📁 File Structure

```
RawCaption/
├── captioners.py              # QuantizedGemmaCaptioner class
├── inference_local.py         # Main inference script
├── json_utils.py              # JSON read/save utilities
├── log.py                     # Comprehensive logging utilities
├── requirements_local.yml     # Environment dependencies
├── run_local.sh              # Shell script for easy execution
├── setup_quantized.sh        # Setup script for CUDA compilation
└── README.md                 # This file
```

## 🚀 Local Pipeline

- **Model**: Google Gemma 3 27B (quantized Q4_0)
- **Framework**: llama-cpp-python with CUDA support
- **Script**: `inference_local.py`
- **Shell Script**: `run_local.sh`

## 🔧 Setup

### Environment Setup

1. **Create conda environment**:
```bash
conda env create -f requirements_local.yml
conda activate gemma-captioning
```

2. **Setup CUDA support**:
```bash
# Run setup script to compile llama-cpp-python with CUDA
./setup_quantized.sh
```

### Model Download

The pipeline will automatically download the model on first run, or you can manually download it:

**⚠️ Security Note**: Never commit your HuggingFace token to version control!

```bash
# Option 1: Set environment variable (Recommended)
export HF_TOKEN="your_huggingface_token_here"

# Option 2: Create .env file (copy from env_example.txt)
cp env_example.txt .env
# Edit .env and add your actual token

# Download quantized model (27B parameters, ~10.7GB)
python -c "
from captioners import download_model_if_needed
download_model_if_needed()
"
```

**Note**: You'll need a HuggingFace account and token to download the model. Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

## 📊 Features

### Comprehensive Logging
- 🎯 **Progress tracking** with emoji indicators
- 💾 **Auto-save** every 30 seconds
- 🔄 **Resume capability** from interruptions
- 📈 **Resource monitoring** (GPU memory, CPU usage)
- ⚠️ **Error handling** with detailed logging
- 🎮 **Performance metrics** (speed, throughput)

### Multi-threading Support
- **Configurable workers** (default: 4 threads)
- **Thread-safe** progress saving
- **Graceful interruption** handling

### Memory Management
- **GPU memory monitoring** before/after processing
- **Automatic cleanup** on completion
- **Device auto-detection** (CUDA/CPU)

## 🏃‍♂️ Usage

```bash
# Basic usage (auto-downloads model)
./run_local.sh input.json output.json

# With custom parameters
python inference_local.py \
    --input input.json \
    --output output.json \
    --n-gpu-layers -1 \
    --max-workers 4 \
    --image-dir query_image
```

## 📋 Command Line Arguments

- `--input, -i`: Input JSON file path
- `--output, -o`: Output JSON file path
- `--image-dir, -img`: Directory containing images (default: "query_image")
- `--start-idx, -s`: Starting index for processing (default: 0)
- `--max-workers, -w`: Maximum worker threads (default: 4)
- `--log, -l`: Log file path
- `--model-path, -m`: Path to GGUF model file (auto-download if not provided)
- `--n-gpu-layers, -g`: GPU layers to offload (-1 for all, default: -1)
- `--model-name`: Model filename for download
- `--repo-id`: HuggingFace repository ID

## 📝 Input/Output Format

### Input JSON Format
```json
[
    {
        "retrieved_image_id1": "image_001",
        "image_raw_caption": ""
    },
    {
        "retrieved_image_id1": "image_002", 
        "image_raw_caption": ""
    }
]
```

### Output JSON Format
```json
[
    {
        "retrieved_image_id1": "image_001",
        "image_raw_caption": "A beautiful sunset over mountains with golden light reflecting on the peaks."
    },
    {
        "retrieved_image_id1": "image_002",
        "image_raw_caption": "A modern office space with clean desks and natural lighting."
    }
]
```

## 🔍 Logging Examples

```
🚀 Starting Gemma 3 27B captioning pipeline...
📊 System Info: CUDA available, 24GB GPU memory
📥 Loading model: Quantized Gemma 3 27B
⏱️ Model loaded in 29.2 seconds
📈 Initial GPU memory: 2.1GB / 24GB
📄 Loaded 1000 entries from JSON
🔄 Resuming from 150 processed entries
🔄 Started periodic saver thread
📋 Found 850 queries to process
✅ [1/850] image_001.jpg: "A scenic mountain landscape with snow-capped peaks"
💾 Auto-save #1: 45 processed, 0 errors
📈 GPU memory after captioning: 8.2GB / 24GB
```

## ⚡ Performance

- **Model Size**: 27B parameters (quantized Q4_0)
- **Memory Usage**: ~8-12GB GPU memory
- **Speed**: Fast inference with CUDA acceleration
- **Quality**: Very high accuracy with quantized model
- **Setup**: Simple setup with automatic model download

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce `--max-workers` to 1-2
   - Reduce `--n-gpu-layers` (try 20, 10, or 5)
   - Monitor GPU memory usage in logs

2. **Model download fails**:
   - Check internet connection
   - Ensure sufficient disk space (~10.7GB)
   - Set your HF_TOKEN: `export HF_TOKEN="your_token_here"`
   - Get token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Try manual download with `huggingface-cli`

3. **Import errors**:
   - Ensure conda environment is activated
   - Reinstall requirements: `conda env update -f requirements_local.yml`

### Performance Tips

- **GPU memory**: Monitor with `nvidia-smi`
- **Threading**: Adjust `--max-workers` based on your system
- **GPU layers**: Use `--n-gpu-layers -1` for maximum GPU utilization
- **Resume**: Use `--start-idx` to resume from specific point

## 📚 Dependencies

### Core Dependencies
- `llama-cpp-python` (CUDA compiled)
- `huggingface-hub`
- `numpy`

### Logging Dependencies
- `psutil` (system monitoring)
- `GPUtil` (GPU monitoring)

## 🔄 Migration from API

To switch from API-based captioning to local:

1. **Replace API calls** with local inference script
2. **Update input/output** paths
3. **Configure logging** for your environment
4. **Test with small dataset** first
5. **Monitor resource usage** during processing

## 📞 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Verify environment setup
3. Test with minimal dataset
4. Review GPU memory usage