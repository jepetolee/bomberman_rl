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


def create_model_from_config(config: Dict[str, Any], device=None, strict_yaml=True):
    """
    Create model from configuration
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        strict_yaml: If True, require all parameters to be explicitly set in YAML (no defaults)
    
    Returns:
        Initialized model (PolicyValueViT or PolicyValueEfficientGTrXL)
    """
    import torch
    import settings as s
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_cfg = get_model_config(config)
    model_type = model_cfg.get('type', 'vit').lower()
    
    if model_type == 'recursive_gtrxl':
        # Recursive GTrXL: Single block used recursively
        from agent_code.ppo_agent.models.efficient_gtrxl import PolicyValueRecursiveGTrXL
        rec_cfg = model_cfg.get('recursive_gtrxl', {})
        
        # Strict mode: require all parameters in YAML
        if strict_yaml:
            required_keys = ['cnn_base_channels', 'cnn_width_mult', 'n_layers_simulated', 'num_heads', 'memory_size']
            missing = [k for k in required_keys if k not in rec_cfg]
            if missing:
                raise ValueError(f"YAML에 필수 RecursiveGTrXL 설정이 없습니다: {missing}. "
                               f"config/trm_config.yaml의 model.recursive_gtrxl 섹션에 명시해야 합니다.")
            if 'embed_dim' not in model_cfg:
                raise ValueError("YAML에 embed_dim이 없습니다. config/trm_config.yaml의 model 섹션에 명시해야 합니다.")
            if 'in_channels' not in model_cfg:
                raise ValueError("YAML에 in_channels가 없습니다. config/trm_config.yaml의 model 섹션에 명시해야 합니다.")
            if 'num_actions' not in model_cfg:
                raise ValueError("YAML에 num_actions가 없습니다. config/trm_config.yaml의 model 섹션에 명시해야 합니다.")
        
        model = PolicyValueRecursiveGTrXL(
            in_channels=model_cfg['in_channels'] if strict_yaml else model_cfg.get('in_channels', 10),
            num_actions=model_cfg['num_actions'] if strict_yaml else model_cfg.get('num_actions', 6),
            img_size=tuple(model_cfg.get('img_size', [s.ROWS, s.COLS])),
            # CNN parameters
            cnn_base_channels=rec_cfg['cnn_base_channels'] if strict_yaml else rec_cfg.get('cnn_base_channels', 32),
            cnn_width_mult=rec_cfg['cnn_width_mult'] if strict_yaml else rec_cfg.get('cnn_width_mult', 1.0),
            # Transformer parameters
            embed_dim=model_cfg['embed_dim'] if strict_yaml else model_cfg.get('embed_dim', 256),
            n_layers_simulated=rec_cfg['n_layers_simulated'] if strict_yaml else rec_cfg.get('n_layers_simulated', 4),
            num_heads=rec_cfg['num_heads'] if strict_yaml else rec_cfg.get('num_heads', 8),
            mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            dropout=model_cfg.get('dropout', 0.0),
            # Memory
            memory_size=rec_cfg['memory_size'] if strict_yaml else rec_cfg.get('memory_size', 256),
        ).to(device)
    elif model_type == 'efficient_gtrxl':
        # EfficientNetB0 + GTrXL model
        from agent_code.ppo_agent.models.efficient_gtrxl import PolicyValueEfficientGTrXL
        eff_cfg = model_cfg.get('efficient_gtrxl', {})
        
        # Strict mode: require all parameters in YAML
        if strict_yaml:
            required_keys = ['cnn_base_channels', 'cnn_width_mult', 'gtrxl_depth', 'num_heads', 'memory_size']
            missing = [k for k in required_keys if k not in eff_cfg]
            if missing:
                raise ValueError(f"YAML에 필수 EfficientGTrXL 설정이 없습니다: {missing}. "
                               f"config/trm_config.yaml의 model.efficient_gtrxl 섹션에 명시해야 합니다.")
            if 'embed_dim' not in model_cfg:
                raise ValueError("YAML에 embed_dim이 없습니다. config/trm_config.yaml의 model 섹션에 명시해야 합니다.")
            if 'in_channels' not in model_cfg:
                raise ValueError("YAML에 in_channels가 없습니다. config/trm_config.yaml의 model 섹션에 명시해야 합니다.")
            if 'num_actions' not in model_cfg:
                raise ValueError("YAML에 num_actions가 없습니다. config/trm_config.yaml의 model 섹션에 명시해야 합니다.")
        
        model = PolicyValueEfficientGTrXL(
            in_channels=model_cfg['in_channels'] if strict_yaml else model_cfg.get('in_channels', 10),
            num_actions=model_cfg['num_actions'] if strict_yaml else model_cfg.get('num_actions', 6),
            img_size=tuple(model_cfg.get('img_size', [s.ROWS, s.COLS])),
            # CNN parameters
            cnn_base_channels=eff_cfg['cnn_base_channels'] if strict_yaml else eff_cfg.get('cnn_base_channels', 32),
            cnn_width_mult=eff_cfg['cnn_width_mult'] if strict_yaml else eff_cfg.get('cnn_width_mult', 1.0),
            # Transformer parameters
            embed_dim=model_cfg['embed_dim'] if strict_yaml else model_cfg.get('embed_dim', 512),
            gtrxl_depth=eff_cfg['gtrxl_depth'] if strict_yaml else eff_cfg.get('gtrxl_depth', 4),
            num_heads=eff_cfg['num_heads'] if strict_yaml else eff_cfg.get('num_heads', 8),
            mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            dropout=model_cfg.get('dropout', 0.0),
            # Memory
            memory_size=eff_cfg['memory_size'] if strict_yaml else eff_cfg.get('memory_size', 256),
        ).to(device)
    else:
        # ViT Only model (default)
        from agent_code.ppo_agent.models.vit import PolicyValueViT
        patch_cfg = model_cfg.get('patch', {})
        vit_cfg = model_cfg.get('vit', {})
        
        model = PolicyValueViT(
            in_channels=model_cfg.get('in_channels', 10),
            num_actions=model_cfg.get('num_actions', 6),
            img_size=tuple(model_cfg.get('img_size', [s.ROWS, s.COLS])),
            embed_dim=model_cfg.get('embed_dim', 512),  # Increased default
            depth=vit_cfg.get('depth', 6),  # Increased default
            num_heads=vit_cfg.get('num_heads', 8),  # Increased default
            mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            drop=model_cfg.get('dropout', 0.0),
            attn_drop=0.0,
            patch_size=patch_cfg.get('size', 1),
            use_cls_token=vit_cfg.get('use_cls_token', False),
            mixer=vit_cfg.get('mixer', 'attn'),
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

