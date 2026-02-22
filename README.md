# Healthcare Assistant via LLM Fine-Tuning

A domain-specific healthcare chatbot built by fine-tuning TinyLlama-1.1B using LoRA (Low-Rank Adaptation) on medical question-answer pairs.

## Links

- **Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1HR8HoNGk3ArAkCIKHsLu4WKbXjJYPH67?usp=sharing)
- **Demo Video:** [Watch Demo Video](https://youtu.be/I0VVHOadkWg)

## Project Overview

This project implements a healthcare assistant capable of answering medical questions about diseases, symptoms, treatments, and medical concepts. The assistant is built by fine-tuning a pre-trained Large Language Model using parameter-efficient techniques optimized for Google Colab's free GPU resources.

### Key Features

- Fine-tuned on 3000+ medical flashcard question-answer pairs
- Parameter-efficient training using LoRA with 4-bit quantization
- Multiple hyperparameter experiments documented
- Comprehensive evaluation using BLEU, ROUGE scores
- Interactive web interface using Gradio
- Complete comparison between base and fine-tuned models

## Quick Start

### Running on Google Colab

Click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HR8HoNGk3ArAkCIKHsLu4WKbXjJYPH67?usp=sharing)

### Local Setup

1. Clone the repository:

```bash
git clone https://github.com/aubert-gloire/Domain-Specific-Assistant-via-LLMs-Fine-Tuning.git
cd YOUR_REPO
```

1. Install dependencies:

```bash
pip install transformers datasets peft trl accelerate bitsandbytes gradio rouge-score sacrebleu sentencepiece protobuf torch
```

1. Run the notebook in Jupyter or upload to Google Colab

## Dataset

**Source:** Medical Meadow Medical Flashcards from Hugging Face

**Dataset ID:** `medalpaca/medical_meadow_medical_flashcards`

**Size:** 3000 training samples, 300 evaluation samples

**Format:** Question-answer pairs covering various medical topics including:

- Disease definitions and pathophysiology
- Symptoms and diagnosis
- Treatment protocols
- Medical terminology
- Anatomy and physiology

### Data Preprocessing

The dataset undergoes comprehensive preprocessing:

- Text cleaning and normalization using regex
- Removal of extra whitespace and special characters
- Tokenization using TinyLlama's tokenizer
- Formatting into instruction-response templates
- Sequence length management (max 512 tokens)
- Train-test split (90/10)

## Model Architecture

**Base Model:** TinyLlama-1.1B-Chat-v1.0

**Fine-tuning Method:** LoRA (Low-Rank Adaptation)

### LoRA Configuration

```python
LoraConfig(
    r=16,                    # Rank of low-rank matrices
    lora_alpha=32,          # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Quantization

4-bit quantization using bitsandbytes for memory efficiency:

- Quantization type: NF4 (Normal Float 4)
- Compute dtype: float16
- Double quantization enabled

**Result:** Only 0.5-1% of parameters are trainable, enabling efficient training on limited GPU resources.

## Training Details

### Hyperparameter Experiments

Four experiments were conducted to optimize performance:

| Experiment | Learning Rate | Batch Size | Epochs | LoRA Rank | Training Time | Final Loss |
|------------|--------------|------------|--------|-----------|---------------|------------|
| Exp 1 | 2e-4 | 4 | 1 | 16 | ~15 min | Baseline |
| Exp 2 | 5e-5 | 4 | 2 | 16 | ~30 min | Lower LR test |
| Exp 3 | 2e-4 | 2 | 1 | 8 | ~20 min | Resource-constrained |
| Exp 4 | 1e-4 | 4 | 1 | 32 | ~18 min | Higher capacity |

### Training Configuration

```python
TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    max_grad_norm=0.3
)
```

### GPU Requirements

- GPU: T4 (Google Colab free tier)
- Memory usage: ~8-10 GB
- Training time: ~15-30 minutes per experiment

## Evaluation Metrics

### Quantitative Metrics

- **BLEU Score:** Measures n-gram overlap between predicted and reference answers
- **ROUGE-1:** Unigram overlap
- **ROUGE-2:** Bigram overlap
- **ROUGE-L:** Longest common subsequence

### Qualitative Testing

Sample interactions demonstrate the model's ability to:

- Define medical terms accurately
- Explain disease processes
- List symptoms and treatments
- Handle medical terminology appropriately

### Base Model vs Fine-tuned Comparison

The notebook includes side-by-side comparisons showing:

- Fine-tuned model provides more accurate medical information
- Base model responses are generic and less domain-specific
- Fine-tuned model uses appropriate medical terminology
- Improved relevance and coherence in healthcare context

## Deployment

### Gradio Web Interface

The model is deployed using Gradio, providing an intuitive interface with:

- Text input for medical questions
- Adjustable temperature for response creativity
- Configurable maximum response length
- Example questions for quick testing
- Shareable public link for remote access

### Running the Interface

```python
demo.launch(share=True)
```

This generates a public URL that can be shared for testing.

## Example Interactions

**Question:** What is hypertension?

**Response:** Hypertension is a medical condition characterized by persistently elevated blood pressure in the arteries. It is defined as systolic blood pressure above 140 mmHg or diastolic pressure above 90 mmHg. Chronic hypertension can lead to serious health complications including heart disease, stroke, and kidney damage.

**Question:** What are the symptoms of diabetes?

**Response:** Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Type 1 diabetes symptoms often develop quickly, while Type 2 diabetes symptoms develop more gradually.

## Project Structure

```
├── healthcare_assistant_finetuning.ipynb  # Main notebook
├── README.md                              # This file                        

```

## Key Insights

### Impact of Fine-tuning

1. **Domain Specialization:** The model successfully adapted from general conversation to healthcare-specific responses
2. **Accuracy Improvement:** Fine-tuned model demonstrates better understanding of medical concepts
3. **Terminology Usage:** Appropriate use of medical terminology compared to base model
4. **Response Quality:** More structured and informative answers to healthcare queries

### Hyperparameter Impact

- **Learning Rate:** Higher learning rates (2e-4) converge faster but may be less stable
- **LoRA Rank:** Higher ranks provide more capacity but increase training time
- **Epochs:** Single epoch sufficient for good results given dataset quality
- **Batch Size:** Larger batches improve stability with gradient accumulation

## Requirements

### Python Packages

- transformers >= 4.36.0
- datasets >= 2.16.0
- peft >= 0.7.0
- trl >= 0.7.0
- accelerate >= 0.25.0
- bitsandbytes >= 0.41.0
- gradio >= 4.11.0
- rouge-score >= 0.1.2
- sacrebleu >= 2.3.1
- sentencepiece >= 0.1.99
- protobuf >= 4.25.0
- torch >= 2.1.0

### Hardware Requirements

- GPU with at least 8 GB VRAM (T4 or better)
- 12 GB RAM minimum
- 10 GB disk space for models and datasets

## Dataset Citation

```
@misc{medical_meadow,
  author = {Medical AI Team},
  title = {Medical Meadow Medical Flashcards},
  year = {2023},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or collaboration:

- Email: <a.bihibindi1@alustudent.com>
