"""
One-shot project structure creation script
Run: python setup_project.py
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Define the complete structure
STRUCTURE = {
    "cpp": {
        "include": ["types.hpp", "order_book.hpp", "order_matcher.hpp", "data_loader.hpp"],
        "src": ["order_book.cpp", "order_matcher.cpp", "data_loader.cpp", "bindings.cpp"],
        "tests": ["test_order_book.cpp", "test_matcher.cpp"],
        "_files": ["CMakeLists.txt"]
    },
    "python": {
        "data": ["__init__.py", "download_data.py", "process_l2.py", "data_loader.py"],
        "env": ["__init__.py", "market_env.py", "state.py", "rewards.py"],
        "agents": ["__init__.py", "base_agent.py", "ppo_agent.py", "sac_agent.py", "random_agent.py"],
        "baselines": ["__init__.py", "avellaneda_stoikov.py", "simple_mm.py"],
        "training": ["__init__.py", "train_ppo.py", "train_sac.py", "config.py"],
        "backtesting": ["__init__.py", "backtest.py", "metrics.py"],
        "live_trading": ["__init__.py", "binance_connector.py", "paper_trader.py", "order_manager.py", "logger.py"],
        "utils": ["__init__.py", "plotting.py", "analysis.py"],
        "_files": ["__init__.py"]
    },
    "data": {
        "raw": [".gitkeep"],
        "processed": [".gitkeep"],
        "logs": [".gitkeep"]
    },
    "models": {
        "ppo": [".gitkeep"],
        "sac": [".gitkeep"]
    },
    "notebooks": [
        "01_data_exploration.ipynb",
        "02_baseline_performance.ipynb",
        "03_rl_training.ipynb",
        "04_live_results_analysis.ipynb",
        "05_comparison.ipynb"
    ],
    "configs": [
        "env_config.yaml",
        "training_config.yaml",
        "live_config.yaml"
    ],
    "scripts": [
        "build_cpp.sh",
        "download_data.sh",
        "run_live.sh"
    ]
}

def create_structure(base_path: Path, structure: dict, parent_key: str = ""):
    """Recursively create folders and files"""
    for key, value in structure.items():
        if key == "_files":
            # Create files in current directory
            for filename in value:
                filepath = base_path / filename
                filepath.touch()
                print(f"✓ Created file: {filepath.relative_to(BASE_DIR)}")
        elif isinstance(value, dict):
            # Create subdirectory
            folder_path = base_path / key
            folder_path.mkdir(exist_ok=True)
            print(f"✓ Created folder: {folder_path.relative_to(BASE_DIR)}/")
            create_structure(folder_path, value, key)
        elif isinstance(value, list):
            # Create folder with files
            folder_path = base_path / key
            folder_path.mkdir(exist_ok=True)
            print(f"✓ Created folder: {folder_path.relative_to(BASE_DIR)}/")
            for filename in value:
                filepath = folder_path / filename
                filepath.touch()
                print(f"  ✓ Created file: {filepath.relative_to(BASE_DIR)}")

def create_root_files():
    """Create root-level files"""
    root_files = {
        "README.md": "# RL Market Maker with C++ Backend\n\nHigh-frequency trading meets deep RL\n",
        "requirements.txt": "",  # Will be filled separately
        "setup.py": "# Setup file for C++ compilation (to be implemented)\n",
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv

# Data
data/raw/*.csv
data/processed/*.bin
data/logs/*.log

# Models
models/**/*.zip
models/**/*.pth

# C++
build/
*.o
*.a

# IDE
.vscode/
.idea/
*.swp

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
"""
    }
    
    for filename, content in root_files.items():
        filepath = BASE_DIR / filename
        # Fix: Use UTF-8 encoding explicitly
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created: {filename}")

def main():
    print("=" * 60)
    print("RL MARKET MAKER - PROJECT SETUP")
    print("=" * 60)
    print()
    
    # Create root files
    print("Creating root files...")
    create_root_files()
    print()
    
    # Create directory structure
    print("Creating directory structure...")
    create_structure(BASE_DIR, STRUCTURE)
    print()
    
    print("=" * 60)
    print("PROJECT STRUCTURE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate venv: .\\venv\\Scripts\\activate")
    print("3. Install requirements: pip install -r requirements.txt")
    print()

if __name__ == "__main__":
    main()