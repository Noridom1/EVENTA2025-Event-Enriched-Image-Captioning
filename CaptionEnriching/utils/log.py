
import torch
import logging
import psutil

def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
    )

def log_resource_usage():
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
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        logging.info(
            f"💻 [GPU {device}] {batch_info} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB"
        )