#!/usr/bin/env python3
"""
Captioner class for local quantized Gemma image captioning.
"""

import os
import time
import base64
import logging
from typing import Optional

from llama_cpp import Llama
from log import *

HF_TOKEN = os.getenv("HF_TOKEN")


class QuantizedGemmaCaptioner:
    def __init__(self, model_path: Optional[str] = None, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        
        log_model_loading(f"Quantized Gemma 3 27B", f"GPU layers: {n_gpu_layers}")
        
        try:
            load_start_time = time.time()
            
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            load_time = time.time() - load_start_time
            log_model_loaded(f"Quantized Gemma 3 27B", load_time)
            
            # Log initial resource usage
            log_resource_usage()
            
        except Exception as e:
            log_error(f"Failed to load model: {e}")
            raise
    
    def _image_to_base64_data_uri(self, image_path: str) -> str:
        """Convert image to base64 data URI."""
        try:
            with open(image_path, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
        except Exception as e:
            log_error(f"Failed to convert image to base64 {image_path}: {e}")
            raise
    
    def generate_caption(self, image_path: str, system_prompt: str, user_prompt: str) -> str:
        try:
            image_data_url = self._image_to_base64_data_uri(image_path)
            
            log_gpu_memory("Before captioning")
            
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}}
                        ]
                    },
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    }
                ],
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9
            )
            
            caption = response['choices'][0]['message']['content']
            
            log_gpu_memory("After captioning")
            
            return caption
            
        except Exception as e:
            log_error(f"Error generating caption for {image_path}: {e}")
            raise


def download_model_if_needed(model_name: str = "gemma-3-27b-it-q4_0.gguf", 
                           repo_id: str = "google/gemma-3-27b-it-qat-q4_0-gguf",
                           local_dir: str = "models") -> str:
    from huggingface_hub import hf_hub_download
    
    model_path = os.path.join(local_dir, model_name)
    
    if not os.path.exists(model_path):
        logging.info(f"📥 Downloading model {model_name} from {repo_id}...")
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=model_name,
                local_dir=local_dir,
                token=HF_TOKEN if HF_TOKEN != "your_huggingface_token_here" else None
            )
            logging.info(f"✅ Model downloaded successfully to {model_path}")
        except Exception as e:
            log_error(f"Failed to download model: {e}")
            raise
    else:
        logging.info(f"✅ Model already exists at {model_path}")
    
    return model_path 