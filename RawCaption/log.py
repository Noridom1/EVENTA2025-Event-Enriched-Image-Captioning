import torch
import logging
import psutil
import time
from pathlib import Path

def setup_logging(log_path):
    """Setup logging with both file and console handlers."""
    # Create log directory if it doesn't exist
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def log_resource_usage():
    """Log current resource usage."""
    # CPU memory
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    logging.debug(f"🧠 Memory Usage: {memory_mb:.2f} MB")

    # GPU (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        logging.debug(f"🟩 GPU Memory Allocated: {gpu_memory:.2f} MB")
        logging.debug(f"GPU Device: {torch.cuda.get_device_name()}")
    else:
        logging.debug("🟥 GPU not available")

def log_gpu_memory(batch_info="", device=0):
    """Log GPU memory usage with emoji indicators."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        logging.info(
            f"💻 [GPU {device}] {batch_info} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB"
        )

def log_captioning_progress(current, total, image_id, caption_preview=""):
    """Log captioning progress with emoji indicators."""
    progress = (current / total) * 100
    caption_preview = caption_preview[:50] + "..." if len(caption_preview) > 50 else caption_preview
    logging.info(f"📸 [{current}/{total}] {progress:.1f}% | Image: {image_id} | Caption: {caption_preview}")

def log_batch_start(batch_num, batch_size, total_remaining):
    """Log batch processing start."""
    logging.info(f"🔄 Starting batch {batch_num} | Batch size: {batch_size} | Remaining: {total_remaining}")

def log_batch_complete(batch_num, successful, failed, avg_time):
    """Log batch processing completion."""
    logging.info(f"✅ Batch {batch_num} completed | Success: {successful} | Failed: {failed} | Avg time: {avg_time:.2f}s")

def log_error(error_msg, image_id=""):
    """Log errors with emoji indicator."""
    if image_id:
        logging.error(f"❌ Error processing {image_id}: {error_msg}")
    else:
        logging.error(f"❌ Error: {error_msg}")

def log_warning(warning_msg, image_id=""):
    """Log warnings with emoji indicator."""
    if image_id:
        logging.warning(f"⚠️ Warning for {image_id}: {warning_msg}")
    else:
        logging.warning(f"⚠️ Warning: {warning_msg}")

def log_save_progress(save_count, total_processed, errors):
    """Log progress saving."""
    logging.info(f"💾 Auto-saving progress | Processed: {total_processed} | Errors: {errors} | Save #{save_count}")

def log_model_loading(model_name, device):
    """Log model loading with emoji indicators."""
    logging.info(f"🤖 Loading model: {model_name}")
    logging.info(f"🖥️ Device: {device}")

def log_model_loaded(model_name, load_time):
    """Log successful model loading."""
    logging.info(f"✅ Model loaded successfully: {model_name} in {load_time:.2f}s")

def log_processing_start(total_images, start_idx, max_workers):
    """Log processing start with configuration."""
    logging.info(f"🚀 Starting image captioning job...")
    logging.info(f"📊 Total images: {total_images}")
    logging.info(f"📍 Starting index: {start_idx}")
    logging.info(f"👥 Max workers: {max_workers}")

def log_processing_complete(total_processed, total_errors, duration):
    """Log processing completion."""
    logging.info(f"🎉 Processing completed!")
    logging.info(f"📈 Total processed: {total_processed}")
    logging.info(f"❌ Total errors: {total_errors}")
    logging.info(f"⏱️ Duration: {duration / 60:.2f} minutes")

def log_resume_detected(resume_count):
    """Log when resuming from previous run."""
    logging.info(f"🔄 Resuming from previous run | {resume_count} images already processed")

def log_memory_cleanup():
    """Log memory cleanup operations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.debug("🧹 GPU cache cleared")
    logging.debug("🧹 Memory cleanup completed") 