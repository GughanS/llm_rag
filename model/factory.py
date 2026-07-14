import os
from typing import Dict, Any
import safetensors.torch
from model.config import TransformerConfig
from model.transformer import Transformer

class ModelFactory:
    """Factory pattern for instantiating and loading the Transformer model.
    
    Centralizes all initialization logic, configuration parsing, and secure
    weight loading (enforcing safetensors) in one place.
    """
    
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> Transformer:
        """Create a fresh model from a configuration dictionary."""
        config = TransformerConfig(**config_dict)
        return Transformer(config)
        
    @staticmethod
    def load_from_checkpoint(checkpoint_dir: str) -> Transformer:
        """Load a model from a checkpoint directory.
        
        Strictly enforces safetensors for security. torch.load (pickle) is not allowed.
        """
        config_path = os.path.join(checkpoint_dir, "config.json")
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                f"Note: Only safetensors format is supported for security reasons."
            )
            
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            
        config = TransformerConfig(**config_dict)
        model = Transformer(config)
        
        # Securely load weights
        safetensors.torch.load_model(model, model_path)
        
        return model
