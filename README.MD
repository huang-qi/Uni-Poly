# Uni-Poly

Uni-Poly is a novel framework that integrates diverse data modalities to achieve a comprehensive and unified representation of polymers for property prediction. By combining multiple polymer representations with domain-specific knowledge, it advances the state-of-the-art in polymer material analysis and discovery.

## Project Overview

Uni-Poly addresses the limitations of conventional polymer property prediction approaches by integrating:

- **Multiple Molecular Representations:**
  - SMILES sequences
  - 2D molecular graphs
  - 3D molecular geometries
  - Polymer fingerprints
  - Domain-specific textual descriptions

- **Key Features:**
  - Comprehensive multi-modal polymer representation
  - Integration of domain-specific knowledge through text
  - Superior performance across various property prediction tasks

Our experimental results demonstrate that Uni-Poly outperforms traditional single-modality and restricted multi-modality methods, highlighting the transformative potential of leveraging multi-modal and domain-specific information in polymer science.

## Directory Structure

```
Uni-Poly/
├── src/                    # Source code directory
│   ├── dataset/           # Dataset processing modules
│   │   ├── geom_data.py   # Geometric data processing
│   │   ├── graph_data.py  # Graph data structures
│   │   ├── dataset.py     # Core dataset implementations
│   │   └── dataloader.py  # Data loading utilities
│   ├── modules/           # Core model modules
│   │   ├── geom.py       # Geometric processing modules
│   │   └── ...           # Other model modules
│   ├── utils.py          # Utility functions
│   └── __init__.py       # Package initialization
├── scripts/               # Training and execution scripts
│   ├── train.py          # Main training script
│   └── pretrain.py       # Pre-training script
├── data/                  # Data directory
│   ├── raw/              # Raw data files (CSV format)
│   │   ├── smi_tg.csv    # SMILES and Tg data
│   │   └── ...          # Other property data files
│   └── smiles_text_dict.json  # SMILES to text mapping
├── pretrained_models/     # Pre-trained model weights
│   └── encoders/         # Pre-trained encoder weights
└── caption_generation/    # Caption generation utilities
    ├── Poly-caption.ipynb # Caption generation notebook
    └── json_input.json   # Example input format
```

## Prerequisites

Before running the project, ensure you have:

1. Python 3.8 or higher
2. CUDA compatible GPU (recommended)
3. Required data files and pre-trained weights (see Data Preparation section)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Uni-Poly.git
cd Uni-Poly
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Property Data Files:
   - Place your polymer property data files in `data/raw/`
   - Required format: CSV files with SMILES and property columns
   - Example: `smi_tg.csv` for glass transition temperature data
   - File format reference available in `data/raw/smi_tg.csv`

2. Pre-trained Weights:
   - Download the pre-trained encoder weights from each repository and place them in the `pretrained_models/encoders/` directory
   - Refer to SI in our paper for detailed model weights information

3. Caption Generation Input:
   - Prepare your input JSON file to generate captions for polymers following the format in `caption_generation/json_input.json`

## Usage

### Caption Generation

Run the caption generation:
   - Navigate to the `caption_generation` directory
   - Run `Poly-caption.ipynb` in Jupyter Notebook

### Pre-training

To run pre-training with default settings:
```bash
bash scripts/pretrain.sh
```

### Training

To train the model:
```bash
bash scripts/train.sh  
```
for advanced options, please refer to the script.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback, please contact:

**QI HUANG**  
Email: huangqi@mail.sim.ac.cn
