# HKM PoC - Holographic Knowledge Manifold Proof of Concept

## Overview
This repository contains the implementation of a Holographic Knowledge Manifold (HKM) that combines neural networks with graph-based knowledge representations for enhanced language understanding and reasoning.

## Project Structure
```
C:/hkm_pipeline/
├── data/           # Datasets (WikiText, FB15k-237)
├── models/         # Trained models and checkpoints
├── scripts/        # Python scripts for each phase
├── outputs/        # Generated outputs and results
├── notebooks/      # Jupyter notebooks for experiments
├── hkm_env/       # Python virtual environment
└── requirements.txt # Python dependencies
```

## Setup Completed
- **Hardware**: Windows 11, 32GB RAM, RTX 4060 8GB VRAM
- **Python**: 3.11.9 with virtual environment
- **Core Libraries**: PyTorch 2.5.1+cu121, transformers, datasets, torch_geometric
- **ML Libraries**: numpy, scipy, scikit-learn, matplotlib
- **Additional**: avalanche-lib, sentence-transformers, notebook

## Installation
```bash
# Activate virtual environment
source C:/hkm_pipeline/hkm_env/Scripts/activate  # Windows (Git Bash)
# or
C:\hkm_pipeline\hkm_env\Scripts\activate.bat     # Windows (CMD)

# Install dependencies
pip install -r requirements.txt
```

## Phases
1. **Prep** (Complete): Environment setup and initial downloads
2. **Phase 1**: Build Knowledge Graph from FB15k-237
3. **Phase 2**: Create Latent Space from WikiText
4. **Phase 3**: Holographic Mapping between spaces
5. **Phase 4**: Training and optimization
6. **Evaluate**: Performance assessment
7. **Demo**: Interactive demonstration

## License
MIT