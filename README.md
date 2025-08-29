# TreeNet: Hierarchical and Parallel CNN Architectures for Adversarial Robustness

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **TreeNet** is a research framework for exploring **hierarchical (HD-CNN)** and **parallel** CNN architectures designed to enhance **adversarial robustness** and **privacy** in deep learning models. The project implements and compares various tree-structured neural network architectures against traditional flat CNNs.

**Key Features:**
- 🌳 **Hierarchical CNN (HD-CNN)** with soft/hard routing mechanisms
- 🔀 **Parallel CNN architectures** with mixture-of-experts fusion
- 🛡️ **Adversarial training** support (TRADES, MART)
- 📊 **Comprehensive evaluation** on CIFAR-10, CIFAR-10-C, and adversarial benchmarks
- 🎯 **Category-aware architectures** for animal/vehicle classification

---

## ✨ Goals
- **Compare**: flat baseline CNN vs. hierarchical/parallel variants
- **Train** with TRADES / MART and evaluate robustness
- **Measure** clean accuracy, adversarial accuracy, and per-category behavior

---

## 📁 Repository Structure

```
TreeNet/
├── HD-CNN-Final/              # Hierarchical CNN implementation
│   ├── main.py               # Main training script for HD-CNN
│   ├── models.py             # HD-CNN model architectures
│   ├── dataset.py            # Data loading and preprocessing
│   ├── losses.py             # Custom loss functions
│   └── readme.md             # HD-CNN specific documentation
├── PRIME-augmentations/       # PRIME augmentation framework
│   ├── train.py              # Training with PRIME augmentations
│   ├── config/               # Configuration files
│   └── requirements.txt      # PRIME-specific dependencies
├── adversarial_robustness_pytorch/  # Adversarial training framework
│   ├── train-tree.py         # Tree model adversarial training
│   ├── train-par.py          # Parallel model training
│   ├── eval-aa.py            # AutoAttack evaluation
│   ├── eval-rb.py            # Robustness benchmark evaluation
│   └── requirements.txt      # Adversarial training dependencies
├── baseline_train.py         # Baseline flat CNN training
└── README.md                 # This file
```

---

## 🧠 Model Variants

### 1) Flat CNN (Baseline)
- `lightresnet20` (≈175k params)
- Standard 10-class classifier.

### 2) Parallel (2-Experts on Coarse Groups: **animal** / **vehicle**)
- architecture
  - parx :
    - **Animal head** (4+1 classes)
    - **Vehicle head** (6+1 classes)
   
  - par:
    - **root head** (10 classes)(for binary classification)   
    - **Animal head** (4+1 classes)
    - **Vehicle head** (6+1 classes)

- Fusion options:
  - **Independent** training + **Mixture-of-Experts (MoE)** fusion
    - either use conf (1-pred[-1]) or prediction of root head as weight W, final logits = W_animal * animal_logits + W_vehicle * vehicle_logits
  - **Joint** training (final logits supervised) with optional auxiliary losses
  - Weighted or linear-schedule **alpha** for multi-loss

### 3) HD-Tree (Hierarchical)
- **Root (coarse)** → **Subroots/Experts**
- **Soft route**: weighted logits by root probabilities (with small “other -5” encouragement; expand to 10-dim)
- **Hard route**: if-based routing by root predictions
- Supports **pretraining** root/subroots then **finetuning** end-to-end.

> “2 - 4,6 model” refers to **2-way coarse routing** (animal vs vehicle) with **4-class** and **6-class** experts.  
> “10 - 4,6 model” refers to composing experts so that final space is **10 classes** via hierarchical fusion.

---

## 🔧 Environment & Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/heyyyyyyi/Treenet.git
cd Treenet
```

2. **Create and activate conda environment**
```bash
conda create -n treenet python=3.10 -y
conda activate treenet
```

3. **Install dependencies** (choose based on your use case)

For **HD-CNN experiments**:
```bash
# Basic PyTorch and dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn
```

For **PRIME augmentations**:
```bash
cd PRIME-augmentations
pip install -r requirements.txt
# or use conda environment
conda env create -f environment.yml
cd ..
```

For **adversarial training**:
```bash
cd adversarial_robustness_pytorch
pip install -r requirements.txt
# or use conda environment  
conda env create -f environment.yml
cd ..
```

4. **Prepare datasets**
```bash
# Create data directory
mkdir -p data

# CIFAR-10 will be automatically downloaded on first run
# For CIFAR-10-C (corruptions), download from:
# https://zenodo.org/record/2535967
# Extract to: ./data/cifar10c/CIFAR-10-C/
```
 
## 🚀 Training & Evaluation

### 1. HD-CNN Training

Train hierarchical CNN models with spectral clustering:

```bash
cd HD-CNN-Final

# Configure dataset path in dataset.py (set root_path)
# Configure parameters in config.py or main.py:
# - Dataset: CIFAR-10/CIFAR-100
# - Number of cluster centers: 2/9  
# - Number of fine classes: 10/100

python main.py
```

**Important**: Before running, update the following in the code:
- `root_path` in `dataset.py` to point to your data directory
- Dataset selection and cluster parameters in configuration files

### 2. PRIME Augmentation Training

Enhanced training with corruption-aware augmentations:

```bash
cd PRIME-augmentations

python -u train.py \
  --config=config/cifar10_cfg.py \
  --config.model.name=lighttreeresnet20 \
  --config.use_prime=True \
  --config.cc_dir=./data/cifar10c/CIFAR-10-C \
  --config.save_dir=./PRIME/linear_lr/
```

### 3. Adversarial Training

#### Tree Model with TRADES
```bash
cd adversarial_robustness_pytorch

python train-tree.py \
  --data-dir ./data \
  --log-dir ./log_test \
  --desc trades_tree \
  --data cifar10 \
  --batch-size 1024 \
  --model lighttreeresnet20 \
  --num-adv-epochs 100 \
  --adv-eval-freq 10 \
  --beta 6
```

#### Tree Model with MART
```bash
python train-tree.py \
  --data-dir ./data \
  --log-dir ./log_test \
  --desc mart_tree \
  --data cifar10 \
  --batch-size 1024 \
  --model lighttreeresnet20 \
  --num-adv-epochs 100 \
  --adv-eval-freq 10 \
  --beta 6 \
  --mart
```

#### Parallel Models
```bash
# Train parallel architecture (parx)
python train-parx.py \
  --data-dir ./data \
  --log-dir ./log_parx \
  --desc parx_experiment \
  --data cifar10 \
  --batch-size 1024 \
  --model lightresnet20 \
  --num-adv-epochs 100

# Train parallel architecture (par)  
python train-par.py \
  --data-dir ./data \
  --log-dir ./log_par \
  --desc par_experiment \
  --data cifar10 \
  --batch-size 1024 \
  --model lightresnet20 \
  --num-adv-epochs 100
```

#### Baseline Flat CNN
```bash
python train.py \
  --data-dir ./data \
  --log-dir ./log_baseline \
  --desc trades_baseline \
  --data cifar10 \
  --batch-size 1024 \
  --model lightresnet20 \
  --num-adv-epochs 100 \
  --adv-eval-freq 10 \
  --beta 6
```

### 4. Evaluation

#### AutoAttack Evaluation
```bash
cd adversarial_robustness_pytorch

python eval-aa.py \
  --data-dir ./data \
  --log-dir ./log_test \
  --desc test \
  --data cifar10
```

#### Robustness Benchmark
```bash
python eval-rb.py \
  --data-dir ./data \
  --log-dir ./baseline_log/origin \
  --desc origin_10_classifier \
  --data cifar10 \
  --threat Linf
```

#### Category-wise Analysis
```bash
python category_group.py \
  --data-dir ./data \
  --log-dir ./baseline_log/origin \
  --desc origin_10_classifier \
  --data cifar10
```
## 📊 Experimental Results

> **Evaluation Metrics**: Top-1 accuracy on CIFAR-10 (clean accuracy) and adversarial accuracy under various attacks.  
> **Note**: Results are averaged over multiple runs. Exact values may vary depending on random seeds and hyperparameters.

### Performance Summary

Our experiments demonstrate that **hierarchical and parallel architectures** can achieve competitive or superior performance compared to traditional flat CNNs, particularly in terms of clean accuracy while maintaining reasonable adversarial robustness.

---

### Baseline (Flat CNN)

| Model                   | Params   | Clean Acc | Adv Acc |
|--------------------------|----------|-----------|---------|
| lightresnet20 (origin)  | 175,258  | **72.20%**| **35.55%** |

---

### Parallel Network — **animal + vehicle** (2-Experts)

| Run / Setting | Notes                                                                                   | Clean Acc | Adv Acc |
|---------------|-----------------------------------------------------------------------------------------|-----------|---------|
| `init`        | Independent training; animal head trained on all data; MoE using `conf = 1 - pred[-1]` | 72.71%    | 41.00%  |
| `init` (MoE2) | Independent training; MoE: `conf*animal_logits + conf*vehicle_logits`                   | 73.07%    | 39.18%  |
| `CE`          | Joint training with CE on final logits (MoE fusion)                                     | 76.89%    | 39.06%  |
| `init_nopretrain` | Joint CE(final) training, no pretrain, MoE fusion                                   | 75.62%    | 38.42%  |
| `1-1 linear`  | Joint CE(final) + α·CE_animal + α·CE_vehicle; α linear schedule                         | 75.77%    | 38.37%  |

---

### Parallel Network — **root + animal + vehicle** (3-branch)

| Run / Setting     | Notes                                                                                                               | α Schedule        | Clean Acc | Adv Acc |
|-------------------|---------------------------------------------------------------------------------------------------------------------|-------------------|-----------|---------|
| `inti-par3`       | Independent training; root (coarse) all-data; **soft route** with other-5 encouragement, root-weighted logits       | —                 | 74.32%    | 39.64%  |
| `CE-par3`         | Independent pretrains + joint CE(final logits), soft routing, weighted loss                                         | —                 | 77.10%    | 38.93%  |
| `alpha-par3-static` | Joint CE(final) + α·root + α·animal + α·vehicle (weights 1:0.5:0.3:0.3, fixed)                                    | static            | **78.56%**| 32.87%  |
| `alpha-par3-linear` | Same as above, but α varies linearly during training                                                              | linear schedule   | 76.46%    | 33.96%  |

---

### HD-Tree (Hierarchical CNN)

| Run / Setting | Notes                                                                                                       | Clean Acc | Adv Acc |
|---------------|-------------------------------------------------------------------------------------------------------------|-----------|---------|
| `pretrain coarse` + `pretrain subroots` + `finetune` | Root trained on all data, subroot1 trained on animal-only, subroot2 on vehicle-only; soft/hard routes tested | 74.28%    | 37.37%  |

---

## 🔍 Key Observations

- **Parallel / hierarchical models** often achieve **higher clean accuracy** than baseline (up to 78.56%).  
- **Robust accuracy** gains depend strongly on fusion strategy and loss balancing:
  - Independent → joint training improves clean accuracy.
  - Weighted loss (α schedules) shifts the clean vs. robust tradeoff.
- **HD-Tree** shows promise but requires careful route calibration (soft vs. hard).
- **Category-aware architectures** (animal/vehicle grouping) provide interpretable performance gains.

---

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA compatibility**: Ensure your PyTorch installation matches your CUDA version
2. **Memory issues**: Reduce batch size if encountering OOM errors
3. **Data path errors**: Update `root_path` in `HD-CNN-Final/dataset.py` before training
4. **Missing dependencies**: Install requirements for the specific component you're using

### HD-CNN Specific Notes

From `HD-CNN-Final/readme.md`:
- Configure dataset selection (CIFAR-10/CIFAR-100) in the code
- Adjust cluster centers (2 for CIFAR-10, 9 for CIFAR-100)
- Increase pre-train and test batch steps for final experiments
- Fine-tune parameters `u_t` and `lam` if needed

---

## 📚 References

This project builds upon and integrates several research frameworks:

- **PRIME Augmentations**: [amodas/PRIME-augmentations](https://github.com/amodas/PRIME-augmentations)
- **Adversarial Robustness**: [imrahulr/adversarial_robustness_pytorch](https://github.com/imrahulr/adversarial_robustness_pytorch)
- **HD-CNN**: Hierarchical Deep CNN architecture for image classification

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. **Report bugs** by opening an issue
2. **Suggest enhancements** or new features
3. **Submit pull requests** with improvements
4. **Share your experimental results** and insights

### Development Setup

```bash
# Fork the repository and clone your fork
git clone https://github.com/YOUR_USERNAME/Treenet.git
cd Treenet

# Create a development branch
git checkout -b feature/your-feature-name

# Make your changes and test them
# Submit a pull request
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Author**: He Yi
- **GitHub**: [@heyyyyyyi](https://github.com/heyyyyyyi)
- **Repository**: [heyyyyyyi/Treenet](https://github.com/heyyyyyyi/Treenet)

For questions, suggestions, or collaboration opportunities, please open an issue or reach out through GitHub.

---

## 🌟 Acknowledgments

Special thanks to the authors and maintainers of the foundational frameworks this project builds upon, and to the research community for advancing the field of adversarial robustness and hierarchical neural architectures.