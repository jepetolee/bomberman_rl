#!/usr/bin/env python3
"""
Configuration Loader for TRM Training
======================================

Loads configuration from YAML file and applies to environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config/trm_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def apply_config_to_env(config: Dict[str, Any], prefix: str = ""):
    """
    Apply configuration to environment variables
    
    Args:
        config: Configuration dictionary
        prefix: Optional prefix for environment variable names
    """
    env_vars = config.get('env_vars', {})
    
    for key, value in env_vars.items():
        env_key = f"{prefix}{key}" if prefix else key
        # Convert to string if needed
        os.environ[env_key] = str(value)


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration"""
    return config.get('model', {})


def get_phase1_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Phase 1 configuration"""
    return config.get('phase1', {})


def get_phase2_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Phase 2 configuration"""
    return config.get('phase2', {})


def get_phase3_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Phase 3 configuration"""
    return config.get('phase3', {})


def apply_preset(config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """
    Apply a preset configuration
    
    Args:
        config: Base configuration
        preset_name: Name of preset (e.g., "small", "medium", "large")
    
    Returns:
        Updated configuration dictionary
    """
    presets = config.get('presets', {})
    
    if preset_name not in presets:
        raise ValueError(f"Preset '{preset_name}' not found. Available: {list(presets.keys())}")
    
    preset = presets[preset_name]
    
    # Deep merge preset into config
    def deep_merge(base: dict, update: dict) -> dict:
        """Recursively merge update into base"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    # Merge preset into model config
    if 'model' in preset:
        config['model'] = deep_merge(config.get('model', {}), preset['model'])
    
    # Merge preset into phase configs
    for phase in ['phase1', 'phase2', 'phase3']:
        if phase in preset:
            config[phase] = deep_merge(config.get(phase, {}), preset[phase])
    
    return config


def create_model_from_config(config: Dict[str, Any], device=None):
    """
    Create TRM model from configuration
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
    
    Returns:
        Initialized model
    """
    import torch
    from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM_Hybrid
    import settings as s
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_cfg = get_model_config(config)
    trm_cfg = model_cfg.get('trm', {})
    patch_cfg = model_cfg.get('patch', {})
    vit_cfg = model_cfg.get('vit', {})
    
    # Use hybrid model: ViT backbone + TRM residual
    # For pretraining: use_trm=False (ViT only)
    # For RL: use_trm=True (ViT + TRM)
    model = PolicyValueViT_TRM_Hybrid(
        in_channels=model_cfg.get('in_channels', 11),
        num_actions=model_cfg.get('num_actions', 6),
        img_size=tuple(model_cfg.get('img_size', [s.ROWS, s.COLS])),
        embed_dim=model_cfg.get('embed_dim', 64),
        # ViT backbone params
        vit_depth=vit_cfg.get('depth', 2),
        vit_heads=vit_cfg.get('num_heads', 4),
        vit_mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
        vit_patch_size=patch_cfg.get('size', 1),
        # TRM settings
        trm_n_latent=trm_cfg.get('n_latent', 6),
        trm_mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
        trm_drop=model_cfg.get('dropout', 0.0),
        trm_patch_size=patch_cfg.get('size', 2),
        trm_patch_stride=patch_cfg.get('stride', 1),
        use_ema=trm_cfg.get('use_ema', True),
        ema_decay=trm_cfg.get('ema_decay', 0.999),
    ).to(device)
    
    return model


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and apply TRM configuration')
    parser.add_argument('--config', type=str, default='config/trm_config.yaml',
                       help='Path to config file')
    parser.add_argument('--preset', type=str, default=None,
                       help='Apply preset configuration (small, medium, large, etc.)')
    parser.add_argument('--apply-env', action='store_true',
                       help='Apply configuration to environment variables')
    parser.add_argument('--print', action='store_true',
                       help='Print configuration')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply preset if specified
    if args.preset:
        config = apply_preset(config, args.preset)
        print(f"Applied preset: {args.preset}")
    
    # Apply to environment
    if args.apply_env:
        apply_config_to_env(config)
        print("Configuration applied to environment variables")
    
    # Print config
    if args.print:
        print("\n" + "="*60)
        print("Configuration:")
        print("="*60)
        import json
        print(json.dumps(config, indent=2, default=str))


if __name__ == '__main__':
    main()

