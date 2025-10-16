# Baseline Evaluation for Phi-4-mini-instruct

This directory contains scripts for establishing baseline performance of `unsloth/Phi-4-mini-instruct` on cybersecurity tasks before fine-tuning.

## Overview

- **Dataset**: 12,118 cybersecurity samples split into train/val/test (8:1:1)
- **Model**: `unsloth/Phi-4-mini-instruct` with Unsloth acceleration
- **Evaluations**: CyberMetric (adversarial classification) + DeepEval (vulnerability scanning)

## Directory Structure

```
train/
├── data/
│   ├── train.json (9,694 samples - 80%)
│   ├── val.json (1,211 samples - 10%)
│   └── test.json (1,213 samples - 10%)
├── results/
│   ├── cybermetric_baseline_phi4.json
│   └── deepeval_baseline_phi4.json
├── split_dataset.py
├── eval_cybermetric.py
├── eval_deepeval.py
├── run_baseline.sh
└── README.md
```

## Setup

### 1. Environment Configuration

Create a `.env` file in the project root with your Gemini API key:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 2. Dataset Splitting

The dataset has already been split. If you need to re-split:

```bash
python3 split_dataset.py
```

## Running Evaluations

### Quick Start

Run both evaluations sequentially:

```bash
./run_baseline.sh
```

### Individual Evaluations

#### CyberMetric Evaluation

Tests adversarial response classification using Gemini 2.5-flash as judge:

```bash
python3 eval_cybermetric.py
```

**Categories evaluated:**
- S: Stuck (repetitive responses)
- R: Reject (ethical refusal)
- G: General (cybersecurity knowledge)
- MI: Malicious but Irrelevant
- MN: Malicious Not Effective
- ME: Malicious Effective

#### DeepEval Evaluation

Tests vulnerability to red-teaming attacks:

```bash
python3 eval_deepeval.py
```

**Vulnerabilities tested:**
- CYBERCRIME
- ILLEGAL_ACTIVITIES
- SHELL_INJECTION
- SQL_INJECTION
- SSRF
- BFLA
- BOLA

## Expected Results

### CyberMetric Results
- **Input**: 1,213 test samples
- **Output**: Classification distribution across 6 categories
- **Judge**: Gemini 2.5-flash
- **Expected runtime**: ~30-60 minutes

### DeepEval Results
- **Input**: Red-teaming prompts
- **Output**: Vulnerability scores and attack success rates
- **Expected runtime**: ~10-20 minutes

## Hardware Requirements

- **GPU**: CUDA-compatible GPU with at least 8GB VRAM
- **RAM**: 16GB+ recommended
- **Storage**: 2GB+ for model and data

## Dependencies

All dependencies are managed through the project's `pyproject.toml`:

- `unsloth`: Fast model loading and inference
- `transformers`: Model handling
- `deepeval`: Red-teaming evaluation
- `google-genai`: Gemini API (new unified SDK - migrated from deprecated `google-generativeai`)
- `python-dotenv`: Environment variable loading

**Note**: This project uses the new `google-genai` SDK instead of the deprecated `google-generativeai` package. The migration ensures compatibility with the latest Gemini API features.

## Next Steps

After running baseline evaluations:

1. **Fine-tune** the model on the training set
2. **Re-run evaluations** on the same test set
3. **Compare results** to measure improvement
4. **Analyze** which categories show the most improvement

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `max_seq_length` in evaluation scripts
2. **Gemini API errors**: Check your API key and quota
3. **Model loading issues**: Ensure Unsloth is properly installed

### Logs

Check console output for detailed progress and any error messages. Results are automatically saved to the `results/` directory.

## Performance Notes

- **Unsloth acceleration**: Significantly faster inference than standard transformers
- **4-bit quantization**: Reduces memory usage while maintaining quality
- **Batch processing**: Evaluations run sequentially for stability

## Model Details

- **Base Model**: `unsloth/Phi-4-mini-instruct`
- **Quantization**: 4-bit (load_in_4bit=True)
- **Max Sequence Length**: 2048 tokens
- **Chat Template**: Llama-3.1 format
- **Inference**: Fast inference enabled with Unsloth
