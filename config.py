from pathlib import Path
def get_config():
    return {
        "batch_size": 2,        # Reduced batch size due to longer sequences
        "num_epochs": 5,        
        "lr": 1e-4,             # Fixed: was 10-4 instead of 1e-4
        "seq_len": 320,         # Increased to accommodate longer sentences
        "d_model": 128,         
    "lang_src": "en",
    "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"  # Fixed: was "run/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
