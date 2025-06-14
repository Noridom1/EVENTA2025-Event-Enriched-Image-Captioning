#!/bin/bash

# Local Gemma Image Captioning Script
# This script runs the quantized Gemma 3 27B model locally for image captioning.
# It provides a convenient wrapper around the Python inference script with sensible defaults.

# Default values
INPUT_JSON=""
OUTPUT_JSON=""
IMAGE_DIR="query_image"
MODEL_PATH=""
MODEL_NAME="gemma-3-27b-it-q4_0.gguf"
REPO_ID="google/gemma-3-27b-it-qat-q4_0-gguf"
N_GPU_LAYERS=-1
MAX_WORKERS=4
START_IDX=0
LOG_FILE="logs/captioning.log"
SETUP_ONLY=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] INPUT_JSON OUTPUT_JSON"
    echo ""
    echo "Description:"
    echo "  Run local Gemma 3 27B image captioning on a JSON file containing image IDs."
    echo "  The script will automatically download the model if it doesn't exist."
    echo ""
    echo "Arguments:"
    echo "  INPUT_JSON              Input JSON file path"
    echo "  OUTPUT_JSON             Output JSON file path"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -s, --start-idx NUM     Starting index for processing (default: 0)"
    echo "  -img, --image-dir DIR   Image directory (default: query_image)"
    echo "  -m, --model-path PATH   Path to GGUF model file (will download if not provided)"
    echo "  --model-name NAME       Model filename to download (default: gemma-3-27b-it-q4_0.gguf)"
    echo "  --repo-id ID            HuggingFace repository ID (default: google/gemma-3-27b-it-qat-q4_0-gguf)"
    echo "  -g, --gpu-layers NUM    Number of GPU layers (-1 for all, default: -1)"
    echo "  -w, --workers NUM       Max workers (default: 4)"
    echo "  -l, --log FILE          Log file path (default: logs/captioning.log)"
    echo ""
    echo "Examples:"
    echo "  $0 data.json results.json"
    echo "  $0 -s 100 -w 2 data.json results.json"
    echo "  $0 -g -1 -w 4 -img images data.json results.json"
}

# Function to setup environment
setup_environment() {
    echo "🔧 Setting up quantized Gemma environment..."
    
    # Check CUDA availability
    if ! command -v nvcc &> /dev/null; then
        echo "❌ CUDA compiler (nvcc) not found. Please install CUDA toolkit."
        exit 1
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ NVIDIA driver not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    echo "✅ CUDA environment detected"
    nvidia-smi
    
    # Install llama-cpp-python with CUDA support
    echo "📦 Installing llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
    
    # Install other dependencies
    echo "📦 Installing additional dependencies..."
    pip install huggingface-hub psutil numpy typing-extensions diskcache jinja2
    
    # Test GPU support
    echo "🧪 Testing GPU support..."
    python -c "
import llama_cpp
print('GPU support:', llama_cpp.llama_supports_gpu_offload())
"
    
    echo "✅ Environment setup completed!"
}

# Function to download model
download_model() {
    echo "📥 Downloading quantized model..."
    
    # Create models directory
    mkdir -p models
    
    # Download model using huggingface-cli
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$REPO_ID" "$MODEL_NAME" --local-dir ./models/
    else
        echo "⚠️ huggingface-cli not found, using Python download..."
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$REPO_ID', filename='$MODEL_NAME', local_dir='./models/')
"
    fi
    
    echo "✅ Model download completed!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_JSON="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        -img|--image-dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        -g|--gpu-layers)
            N_GPU_LAYERS="$2"
            shift 2
            ;;
        -w|--workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -s|--start-idx)
            START_IDX="$2"
            shift 2
            ;;
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Setup environment if requested
if [ "$SETUP_ONLY" = true ]; then
    setup_environment
    download_model
    echo "🎉 Setup completed! You can now run the captioning script."
    exit 0
fi

# Check required arguments
if [[ -z "$INPUT_JSON" || -z "$OUTPUT_JSON" ]]; then
    echo "Error: Input and output files are required"
    echo ""
    usage
    exit 1
fi

# Check if input file exists
if [[ ! -f "$INPUT_JSON" ]]; then
    echo "Error: Input file not found: $INPUT_JSON"
    exit 1
fi

# Check if image directory exists
if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "Error: Image directory not found: $IMAGE_DIR"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_JSON")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
fi

# Create log directory if needed
LOG_DIR=$(dirname "$LOG_FILE")
if [[ ! -d "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    echo "Created log directory: $LOG_DIR"
fi

# Check if Python script exists
if [[ ! -f "inference_local_quantized.py" ]]; then
    echo "Error: inference_local_quantized.py not found in current directory"
    exit 1
fi

# Check if log module exists
if [[ ! -f "log.py" ]]; then
    echo "Error: log.py not found in current directory"
    exit 1
fi

# Check if model exists, download if needed
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="models/$MODEL_NAME"
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Model not found, downloading..."
        download_model
    fi
fi

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN environment variable not set."
    echo "   Some models may require authentication to download."
    echo "   Set your token with: export HF_TOKEN='your_token_here'"
    echo ""
fi

# Display configuration
echo "=== Quantized Gemma Captioning Configuration ==="
echo "Input JSON:      $INPUT_JSON"
echo "Output JSON:     $OUTPUT_JSON"
echo "Image Directory: $IMAGE_DIR"
echo "Model Path:      $MODEL_PATH"
echo "GPU Layers:      $N_GPU_LAYERS"
echo "Max Workers:     $MAX_WORKERS"
echo "Start Index:     $START_IDX"
echo "Log File:        $LOG_FILE"
echo "================================================"

# Run the Python script
echo "Starting quantized Gemma captioning..."
python inference_local_quantized.py \
    --input "$INPUT_JSON" \
    --output "$OUTPUT_JSON" \
    --model-path "$MODEL_PATH" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --max-workers "$MAX_WORKERS" \
    --start-idx "$START_IDX" \
    --image-dir "$IMAGE_DIR" \
    --log "$LOG_FILE"

# Check exit status
if [[ $? -eq 0 ]]; then
    echo "Quantized Gemma captioning completed successfully!"
    echo "Results saved to: $OUTPUT_JSON"
    echo "Logs saved to: $LOG_FILE"
else
    echo "Quantized Gemma captioning failed!"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi 