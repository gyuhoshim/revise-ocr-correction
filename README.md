# REVISE: A Framework for Revising OCRed Text in Practical Information Systems with Data Contamination Strategy

This repository contains the official implementation of the REVISE framework for OCR (Optical Character Recognition) error correction using language models with synthetic data contamination strategies.
Paper link: [](url)

## 📖 Overview

REVISE is a comprehensive framework that addresses OCR error correction through:

- **Synthetic Data Contamination**: Simulates various types of OCR errors including character substitution, insertion, deletion, segmentation issues, and layout problems
- **Multi-Error Type Training**: Handles character-level, word-level, and document-level OCR errors
- **Instruction-Following Approach**: Uses chat-based templates for training language models to correct OCR errors
- **Distributed Training**: Supports multi-GPU training with DeepSpeed optimization

## 🚀 Key Features

### OCR Error Types Supported
- **Character-level**: Substitution, insertion, deletion, transposition
- **Word-level**: Word deletion, transposition
- **Segmentation**: Over-segmentation (extra spaces) and under-segmentation (missing spaces)
- **Layout**: Multi-column document reading order errors

### Model Training
- Support for various base models (Llama, Gemma, etc.)
- DeepSpeed integration for efficient distributed training
- Automatic train/validation splitting
- WandB integration for experiment tracking

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPUs
- PyTorch 2.0+

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/gyuhoshim/revise-ocr-correction.git
cd revise-ocr-correction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

### Input Data Format
Your training data should be in JSONL format with the following structure:
```json
{"id": "1", "text": "Your clean text here"}
{"id": "2", "text": "Another clean text sample"}
```

### Data Contamination
Use the contamination script to generate OCR-corrupted versions:

```python
from contaminate import process_jsonl_file

# Process your clean data to add OCR errors
process_jsonl_file(
    input_file='./new_data/clean_data.jsonl',
    output_file='./data/contaminated_data.jsonl',
    total_data_lines=30000
)
```

## 🏃‍♂️ Training

### Quick Start
```bash
# Basic training with default settings
python train.py \
    --base_model="meta-llama/Llama-3.2-1B-Instruct" \
    --data_path="data/contaminated_data.jsonl" \
    --output_dir="./checkpoints" \
    --per_device_train_batch_size=2 \
    --num_epochs=3
```

### Advanced Training with Multiple GPUs
```bash
# Use the provided training script
bash train.sh
```

### Training Parameters
- `--base_model`: HuggingFace model identifier
- `--data_path`: Path to contaminated JSONL training data
- `--output_dir`: Directory to save model checkpoints
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate (default: 3e-4)
- `--cutoff_len`: Maximum sequence length
- `--bf16`: Use bfloat16 precision training

## 🔧 Configuration Files

### DeepSpeed Configuration (`deepspeed.json`)
Optimized for multi-GPU training with ZeRO Stage 3 optimization.

### Character Mapping (`mapping.json`)
Defines character substitution mappings for realistic OCR error simulation.

## 📁 Project Structure

```
├── train.py              # Main training script
├── contaminate.py         # Data contamination utilities
├── evaluate.py           # Model evaluation script
├── example_usage.py      # Usage examples and demonstrations
├── train.sh              # Training script with predefined configurations
├── deepspeed.json        # DeepSpeed configuration
├── mapping.json          # Character substitution mappings
├── requirements.txt      # Python dependencies
├── data/                 # Training data directory
├── new_data/            # Original clean data
└── README.md            # This file
```

## 🎯 Usage Examples

### Quick Start
Run the example script to see the framework in action:
```bash
python example_usage.py
```

### 1. Data Contamination
```python
from contaminate import deletion, insertion, substitution

# Apply different contamination types
text = "This is a sample text."
contaminated = deletion(text)
print(contaminated)  # "Ths is  sample txt."
```

### 2. Model Evaluation
Evaluate your trained model:
```bash
python evaluate.py \
    --model_path="./checkpoints" \
    --test_data="test_data.jsonl" \
    --output="evaluation_results.jsonl"
```

### 3. Model Inference
After training, use your model for OCR correction:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./checkpoints")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Your OCR correction code here
```

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{shim-etal-2025-revise,
    title = "{REVISE}: A Framework for Revising {OCR}ed text in Practical Information Systems with Data Contamination Strategy",
    author = "Shim, Gyuho  and
      Hong, Seongtae  and
      Lim, Heuiseok",
    editor = "Rehm, Georg  and
      Li, Yunyao",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-industry.100/",
    doi = "10.18653/v1/2025.acl-industry.100",
    pages = "1423--1434",
    ISBN = "979-8-89176-288-6",
    abstract = "Recent advances in large language models (LLMs) have significantly improved Document AI, demonstrating remarkable performance on document understanding tasks such as question answering. However, existing approaches primarily focus on solving specific tasks, lacking the capability to structurally organize and systematically manage document information. To address this limitation, we propose Revise, a framework that systematically corrects errors introduced by OCR at the character, word, and structural levels. Specifically, Revise employs a comprehensive hierarchical taxonomy of common OCR errors and a synthetic data generation strategy that realistically simulates such errors to train an effective correction model. Experimental results demonstrate that Revise effectively corrects OCR outputs, enabling more structured representation and systematic management of document contents. Consequently, our method significantly enhances downstream performance in document retrieval and question answering tasks, highlighting the potential to overcome the structural management limitations of existing Document AI frameworks."
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Transformers](https://github.com/huggingface/transformers)
- Distributed training with [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)
