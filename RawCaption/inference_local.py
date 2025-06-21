#!/usr/bin/env python3
"""
Local Gemma Image Captioning Script
"""

import os
import sys
import json
import time
import base64
import logging
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from llama_cpp import Llama

from log import *
from captioners import QuantizedGemmaCaptioner, download_model_if_needed
from json_utils import read_json, save_json


def inference_local_gemma_multi_threads(
    captioner: QuantizedGemmaCaptioner,
    json_path: str,
    save_path: str,
    start_idx: int = 0,
    max_workers: int = 4,
    image_dir: str = "query_image"
) -> None:
    """
    Process multiple images for captioning using quantized Gemma model.
    
    Args:
        captioner: Initialized QuantizedGemmaCaptioner instance
        json_path: Path to input JSON file
        save_path: Path to save output JSON file
        start_idx: Starting index for processing
        max_workers: Number of worker threads
        image_dir: Directory containing images
    """
    # Load data
    try:
        data = read_json(json_path)
        logging.info(f"📄 Loaded {len(data)} entries from JSON")
    except Exception as e:
        log_error(f"Failed to load JSON file: {e}")
        raise
    
    sys_prompt = "Only return plain text with no formatting."
    user_prompt = "Describe this image briefly but with enough detail to convey the main subject and context."
    
    # Check for resume
    already_processed = sum(1 for item in data if item.get('image_raw_caption', '') not in ['', None])
    if already_processed > 0:
        log_resume_detected(already_processed)
    
    # Threading setup
    lock = threading.Lock()
    stop_saving = threading.Event()
    processed_count = 0
    error_count = 0
    save_count = 0
    
    # Log processing start
    total_to_process = len([item for item in data if item.get('image_raw_caption', '') in ['', None]])
    log_processing_start(len(data), start_idx, max_workers)
    
    def process_query(idx: int, query: Dict[str, Any]) -> None:
        """Process a single query."""
        nonlocal processed_count, error_count
        
        try:
            start_time = time.time()
            
            # Check if already processed
            if query.get('image_raw_caption', '') not in ['', None]:
                logging.debug(f"⏭️ Query {idx+1} already has caption, skipping")
                return
            
            # Get image path
            image_id = query.get('retrieved_image_id1')
            if not image_id:
                log_warning(f"Missing retrieved_image_id1", f"query_{idx+1}")
                return
                
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            
            # Check if image exists
            if not os.path.exists(image_path):
                log_warning(f"Image not found", image_id)
                return
            
            # Generate caption
            caption = captioner.generate_caption(image_path, sys_prompt, user_prompt)
            
            # Update data
            with lock:
                query['image_raw_caption'] = caption
                processed_count += 1
            
            elapsed = time.time() - start_time
            
            # Log progress
            log_captioning_progress(processed_count, total_to_process, image_id, caption)
            
        except Exception as e:
            with lock:
                error_count += 1
            log_error(f"Error processing query {idx+1}: {e}")
    
    def periodic_saver() -> None:
        """Periodically save progress."""
        nonlocal save_count
        while not stop_saving.is_set():
            time.sleep(30)  # Save every 30 seconds
            with lock:
                save_count += 1
                log_save_progress(save_count, processed_count, error_count)
                try:
                    save_json(data, save_path)
                    logging.debug("💾 Auto-save completed successfully")
                except Exception as e:
                    log_error(f"Auto-save failed: {e}")
    
    # Start the periodic saver thread
    saver_thread = threading.Thread(target=periodic_saver, daemon=True)
    saver_thread.start()
    logging.info("🔄 Started periodic saver thread")
    
    start_time = time.time()
    
    try:
        # Filter queries to process
        queries_to_process = [
            (idx, query) for idx, query in enumerate(data)
            if (query.get('image_raw_caption', '') == '' or query.get('image_raw_caption', '') is None) 
            and idx >= start_idx
        ]
        
        logging.info(f"📋 Found {len(queries_to_process)} queries to process")
        
        # Process with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_query, idx, query)
                for idx, query in queries_to_process
            ]
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log_error(f"Future execution failed: {e}")
    
    except KeyboardInterrupt:
        logging.warning("⚠️ Received keyboard interrupt, stopping...")
    except Exception as e:
        log_error(f"Unexpected error during processing: {e}")
        raise
    finally:
        # Stop the saver thread and do a final save
        stop_saving.set()
        saver_thread.join(timeout=5)
        
        with lock:
            logging.info("💾 Performing final save...")
            try:
                save_json(data, save_path)
                logging.info("✅ Final save completed successfully")
            except Exception as e:
                log_error(f"Final save failed: {e}")
        
        # Log completion
        duration = time.time() - start_time
        log_processing_complete(processed_count, error_count, duration)
        
        # Cleanup
        log_memory_cleanup()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Local Gemma Image Captioning with Logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input JSON file path"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--model-path", "-m",
        default=None,
        help="Path to GGUF model file (will download if not provided)"
    )
    
    parser.add_argument(
        "--model-name",
        default="gemma-3-27b-it-q4_0.gguf",
        help="Model filename to download"
    )
    
    parser.add_argument(
        "--repo-id",
        default="google/gemma-3-27b-it-qat-q4_0-gguf",
        help="HuggingFace repository ID for model download"
    )
    
    parser.add_argument(
        "--n-gpu-layers", "-g",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 for all)"
    )
    
    parser.add_argument(
        "--image-dir", "-img",
        default="query_image",
        help="Directory containing images"
    )
    
    parser.add_argument(
        "--start-idx", "-s",
        type=int,
        default=0,
        help="Starting index for processing"
    )
    
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=4,
        help="Maximum number of worker threads"
    )
    
    parser.add_argument(
        "--system-prompt",
        default="Only return plain text with no formatting.",
        help="System prompt for the model"
    )
    
    parser.add_argument(
        "--user-prompt",
        default="Describe this image briefly but with enough detail to convey the main subject and context.",
        help="User prompt for the model"
    )
    
    parser.add_argument(
        "--log", "-l",
        default="logs/captioning_quantized.log",
        help="Path to log file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log)
    
    # Validate inputs
    if not os.path.exists(args.input):
        log_error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.image_dir):
        log_error(f"Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Download model if needed
        if args.model_path is None:
            model_path = download_model_if_needed(args.model_name, args.repo_id)
        else:
            model_path = args.model_path
        
        # Initialize captioner
        captioner = QuantizedGemmaCaptioner(
            model_path=model_path,
            n_gpu_layers=args.n_gpu_layers
        )
        
        # Run inference
        inference_local_gemma_multi_threads(
            captioner=captioner,
            json_path=args.input,
            save_path=args.output,
            start_idx=args.start_idx,
            max_workers=args.max_workers,
            image_dir=args.image_dir
        )
        
    except Exception as e:
        log_error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 