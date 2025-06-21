#!/bin/bash

# Quantized Gemma Environment Setup Script
# This script sets up the environment for running quantized Gemma 3 27B

set -e  # Exit on any error

echo "🚀 Setting up Quantized Gemma Environment"
echo "=========================================="

# Check system requirements
echo "🔍 Checking system requirements..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found. Please install CUDA toolkit."
    echo "   Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA driver not found. Please install NVIDIA drivers."
    echo "   Download from: https://www.nvidia.com/drivers"
    exit 1
fi

echo "✅ CUDA environment detected"
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

# Check available disk space
echo "💾 Checking disk space..."
available_space=$(df . | awk 'NR==2 {print $4}')
required_space=$((11 * 1024 * 1024))  # 11GB for model

if [ "$available_space" -lt "$required_space" ]; then
    echo "❌ Insufficient disk space. Need at least 11GB, available: $((available_space / 1024 / 1024))GB"
    exit 1
fi
echo "✅ Sufficient disk space available"

# Install llama-cpp-python with CUDA support
echo "📦 Installing llama-cpp-python with CUDA support..."
echo "   This may take several minutes..."

CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install other dependencies
echo "📦 Installing additional dependencies..."
pip install huggingface-hub psutil numpy typing-extensions diskcache jinja2

# Test GPU support
echo "🧪 Testing GPU support..."
python3 -c "
import llama_cpp
gpu_support = llama_cpp.llama_supports_gpu_offload()
print('✅ GPU support:', gpu_support)
if not gpu_support:
    print('❌ GPU support not available. Check CUDA installation.')
    exit(1)
"

# Create directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p logs

# Download model
echo "📥 Downloading quantized Gemma 3 27B model..."
echo "   Model: gemma-3-27b-it-q4_0.gguf (~10.7GB)"
echo "   Repository: google/gemma-3-27b-it-qat-q4_0-gguf"
echo ""

# Check if model already exists
if [ -f "models/gemma-3-27b-it-q4_0.gguf" ]; then
    echo "✅ Model already exists at models/gemma-3-27b-it-q4_0.gguf"
    echo "   Skipping download..."
else
    echo "📥 Downloading model..."
    echo "   This may take a while depending on your internet connection."
    echo ""
    
    # Download using huggingface-cli
    if [ -n "$HF_TOKEN" ]; then
        echo "   Using HF_TOKEN for authentication..."
        huggingface-cli download google/gemma-3-27b-it-qat-q4_0-gguf gemma-3-27b-it-q4_0.gguf --local-dir ./models/ --token $HF_TOKEN
    else
        echo "   No HF_TOKEN found, downloading without authentication..."
        huggingface-cli download google/gemma-3-27b-it-qat-q4_0-gguf gemma-3-27b-it-q4_0.gguf --local-dir ./models/
    fi
    
    # Alternative Python method
    echo ""
    echo "📥 Alternative download method (Python):"
    echo "python -c \""
    echo "import os"
    echo "from captioners import download_model_if_needed"
    echo "os.environ['HF_TOKEN'] = 'your_token_here'  # Set your token"
    echo "download_model_if_needed()"
    echo "\""
fi

# Test model loading
echo "🧪 Testing model loading..."
python3 -c "
from llama_cpp import Llama
import time

print('Loading model...')
start_time = time.time()
llm = Llama(model_path='models/gemma-3-27b-it-q4_0.gguf', n_gpu_layers=-1, verbose=False)
load_time = time.time() - start_time
print(f'✅ Model loaded successfully in {load_time:.2f} seconds')
print('Model is ready for inference!')
"

# Create example usage script
echo "📝 Creating example usage script..."
cat > example_quantized.py << 'EOF'
#!/usr/bin/env python3
"""
Example usage of the Quantized Gemma Captioner
"""

from inference_local_quantized import QuantizedGemmaCaptioner, download_model_if_needed

def main():
    # Download model if needed
    model_path = download_model_if_needed()
    
    # Initialize captioner
    captioner = QuantizedGemmaCaptioner(
        model_path=model_path,
        n_gpu_layers=-1
    )
    
    # Example image path (replace with your actual image)
    image_path = "query_image/example_image.jpg"
    
    # Generate caption
    system_prompt = "Only return plain text with no formatting."
    user_prompt = "Describe this image briefly but with enough detail to convey the main subject and context."
    
    try:
        caption = captioner.generate_caption(image_path, system_prompt, user_prompt)
        print(f"Generated caption: {caption}")
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with a valid image path.")

if __name__ == "__main__":
    main()
EOF

chmod +x example_quantized.py

echo ""
echo "🎉 Setup completed successfully!"
echo "================================"
echo ""
echo "📋 Next steps:"
echo "1. Test the setup: python3 example_quantized.py"
echo "2. Run captioning: ./run_quantized.sh -i your_data.json -o results.json"
echo "3. Check logs: tail -f logs/captioning_quantized.log"
echo ""
echo "📚 Available scripts:"
echo "  - run_quantized.sh: Main runner script"
echo "  - inference_local_quantized.py: Python inference script"
echo "  - example_quantized.py: Example usage"
echo ""
echo "🔧 Configuration:"
echo "  - Model: models/gemma-3-27b-it-q4_0.gguf"
echo "  - GPU layers: All (-1)"
echo "  - Default workers: 4"
echo ""
echo "💡 Tips:"
echo "  - Adjust --gpu-layers if you have memory issues"
echo "  - Use --workers 1-2 for smaller GPUs"
echo "  - Monitor GPU memory with: nvidia-smi" 