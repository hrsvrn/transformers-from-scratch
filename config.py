from pathlib import Path
def get_config():
    return {
        "batch_size": 4,        # Increased since we're reducing other params
        "num_epochs": 5,        # Reduced for faster testing
        "lr": 10**-4,
        "seq_len": 128,         # Reduced from 256 - this is the key change
        "d_model": 512,         # Reduced from 512 - halves memory usage
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "run/tmodel" 
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)