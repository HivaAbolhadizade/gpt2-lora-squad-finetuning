# Fine-Tuning GPT-2 with LoRA on SQuAD: A Parameter-Efficient Approach for Question Answering

This repository implements and evaluates Low-Rank Adaptation (LoRA) for fine-tuning GPT-2 on a subset of the Stanford Question Answering Dataset (SQuAD v1.1). The project investigates hyperparameter configurations to achieve efficient adaptation while maintaining performance comparable to full fine-tuning.

The work is conducted as part of the Special Topics in Artificial Intelligence (Deep Learning) course, Spring 2025, Shahid Bahonar University of Kerman, under the supervision of Dr. Eftekhari.

According to course guidelines, **all reports must be written in Persian**, even though the current implementation summaries below are presented in English.

## Abstract

Large language models (LLMs) like GPT-2 exhibit strong generalization but require resource-intensive fine-tuning for downstream tasks. This project employs LoRA, a parameter-efficient technique, to adapt GPT-2 for question answering on SQuAD. Experiments explore ranks (r = 4, 8, 16, 32, 64), learning rates (7e-5 to 1e-3), and target modules (attention-focused vs. expanded). The best model (r=4, lr=1e-3, attention modules) achieves an F1 score of 65.33 on validation, using only 0.176% of GPT-2's parameters. Results demonstrate LoRA's efficacy in reducing computational overhead while preserving task performance, with ablation studies highlighting trade-offs in capacity and overfitting.

## Introduction

Fine-tuning pre-trained LLMs is essential for task-specific adaptation but often demands substantial computational resources. LoRA addresses this by updating low-rank matrices in selected layers, minimizing trainable parameters. This repository focuses on applying LoRA to GPT-2 for extractive question answering on SQuAD, a benchmark dataset comprising context-question-answer triples. The study aims to:
- Evaluate LoRA's performance relative to baseline GPT-2.
- Analyze hyperparameter impacts through systematic experiments.
- Provide insights into decoding strategies and failure modes.

SQuAD v1.1 consists of over 100,000 questions on Wikipedia articles. We use a subset (2,000 train, 200 validation, 5 test samples) due to resource constraints. Prompts follow: "Context: {context} Question: {question} Answer: {answer}{eos_token}".

## Related Work

LoRA (Hu et al., 2021) enables efficient fine-tuning by decomposing weight updates into low-rank factors, inspired by intrinsic dimensionality in neural networks. GPT-2 (Radford et al., 2019) is a transformer-based autoregressive model. SQuAD (Rajpurkar et al., 2016) evaluates reading comprehension. Prior works show LoRA's success on GLUE and vision tasks, but this project extends it to question answering with ablation on transformer modules.

## Methodology

LoRA adapts weights W via ΔW = A · B, where A ∈ ℝ^{d×r}, B ∈ ℝ^{r×k}, and r ≪ min(d,k). Only A and B are trained, freezing original weights. In GPT-2, we target attention layers (attn.c_attn, attn.c_proj) for semantic relation capture, with variants including MLP (mlp.c_fc, mlp.c_proj) or query-value projections.

Training uses AdamW optimizer (weight decay 0.01), batch size 8 (via gradient accumulation), 5 epochs, max length 256, and fp16 precision. Evaluation metrics: Exact Match (EM) and F1 score. Decoding strategies: greedy, top-k (k=50), nucleus (p=0.95), temperature (1.3).

75 experiments vary:
- Ranks: 4, 8, 16, 32, 64
- Learning rates: 7e-5, 1e-4, 2e-4, 5e-4, 1e-3
- Modules: attention-only, attention+MLP, query-value+attention

## Experimental Setup

Experiments run on Google Colab with T4 GPU. Base GPT-2 (no fine-tuning) serves as baseline. Full fine-tuning was infeasible due to memory limits.

## Results and Discussion

The baseline achieves F1=16.98, EM=0.0 (greedy decoding). LoRA models outperform, with the top configuration yielding F1=65.33, EM=20.0.

| Configuration | Rank (r) | Learning Rate | Target Modules | F1 (%) | EM (%) |
|---------------|----------|---------------|----------------|--------|--------|
| Best LoRA (Greedy) | 4 | 1e-3 | attn.c_attn, attn.c_proj | 65.33 | 20.0 |
| Baseline GPT-2 (Greedy) | - | - | - | 16.98 | 0.0 |
| LoRA (Top-K, k=50) | 4 | 1e-3 | attn.c_attn, attn.c_proj | 20.0 | 0.0 |
| LoRA (Nucleus, p=0.95) | 4 | 1e-3 | attn.c_attn, attn.c_proj | 0.95 | 0.0 |
| LoRA (Temperature, 1.3) | 4 | 1e-3 | attn.c_attn, attn.c_proj | 6.13 | 0.0 |

### Ablation Studies
- **Rank**: Lower ranks (4-8) balance capacity and regularization, preventing overfitting; higher ranks (32+) increase parameters but risk instability on small data.
- **Learning Rate**: Higher rates (1e-3) accelerate convergence; lower rates (7e-5) enhance stability but prolong training.
- **Modules**: Attention-only suffices for context-question alignment; adding MLP boosts non-linearity but heightens overfitting risk.

Greedy decoding excels for precise answers; sampling introduces variability unsuitable for extractive QA.

### Failure Cases
Analysis of errors reveals:
1. Contextual misalignments (e.g., predicting "Denver Broncos" instead of "Carolina Panthers" for Super Bowl 50 loser).
2. Verbose generations replicating context rather than concise spans.
3. Sensitivity to subtle details (e.g., "blue" vs. "gold" in thematic queries).

These suggest limitations in subset size and prompt design.

### Computational Efficiency
LoRA trains ~221K parameters (0.176% of 124M in GPT-2), with ~3 min/epoch and reduced memory via fp16.

## Conclusion

LoRA enables efficient fine-tuning of GPT-2 for QA, achieving strong performance with minimal resources. Future extensions could use full SQuAD or larger models.

## Repository Structure

- **1. QUESTION**: Course assignment guidelines.
- **Best Model**: Top model checkpoints (r=4, lr=1e-3, attention modules).
- **Code**: Scripts for preprocessing, training, evaluation, and analysis.
- **Experiment Results**: Metrics, loss curves, and plots from 75 runs.
- **REPORT.pdf**: Full report in Persian (12 pages).

## Installation

```bash
git clone https://github.com/yourusername/gpt2-lora-squad-finetuning.git
cd gpt2-lora-squad-finetuning
pip install -r requirements.txt
```

`requirements.txt`:
- transformers==4.28.0
- peft==0.3.0
- datasets==2.11.0
- torch==2.0.0
- evaluate==0.4.0

## Usage

### Data Preprocessing
```bash
python Code/preprocess.py --dataset squad --train_size 2000 --val_size 200 --test_size 5
```

### Training
```bash
python Code/train.py --rank 4 --lr 1e-3 --target_modules attn.c_attn attn.c_proj --epochs 5 --batch_size 4 --gradient_accumulation 2
```

### Evaluation
```bash
python Code/evaluate.py --model_path Best\ Model/ --decoding greedy
```

### Analysis
```bash
python Code/analyze_results.py --results_dir Experiment\ Results/
```

## References

- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
- Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. arXiv:1606.05250.

For inquiries, contact Hiva Abolhadizadeh (Student ID: 400405004).
