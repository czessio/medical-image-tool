#!/usr/bin/env python3
"""
Fix the Config class to use class attribute for DEFAULT_CONFIG.
"""
import os
from pathlib import Path

def fix_config_file():
    """Fix the Config class in config.py."""
    config_path = Path('utils/config.py')
    
    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Fix the load method to use the class attribute
    if "return self.DEFAULT_CONFIG.copy()" in content:
        content = content.replace(
            "return self.DEFAULT_CONFIG.copy()",
            "return Config.DEFAULT_CONFIG.copy()"
        )
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print(f"Fixed config.py to use class attribute for DEFAULT_CONFIG")
        return True
    else:
        print("No need to fix config.py, already using correct attribute access")
        return True

if __name__ == "__main__":
    fix_config_file()