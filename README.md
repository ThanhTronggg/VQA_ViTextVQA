# Visual Question Answering (VQA) for Vietnamese Documents

This project implements a Visual Question Answering (VQA) system tailored for Vietnamese documents, enabling users to ask questions about images (e.g., receipts, signs, posters) and receive accurate answers in Vietnamese. The project leverages advanced deep learning models and is trained and tested on the **ViTextVQA** dataset. It includes three main approaches:

1. **Custom VQA Model**: Combines InternViT for image encoding, PhoBERT for text encoding, and a transformer-based decoder for answer generation (`vqa-define.ipynb`).
2. **Fine-tuned Vintern-1B-v2**: Fine-tunes the Vintern-1B-v2 model using LoRA on the ViTextVQA dataset for enhanced document question answering (`Fine_tune_my_data.ipynb`).
3. **ResNet50 and LSTM Model**: Uses ResNet50 for image feature extraction and LSTM for answer prediction (`cv-internvit-lstm.ipynb`).

The project also includes testing scripts to evaluate the fine-tuned Vintern-1B-v2 model on a test dataset (`test-vintern-finetune.ipynb` and `Test_my_model.ipynb`).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Custom VQA Model](#custom-vqa-model)
  - [Fine-tuning Vintern-1B-v2](#fine-tuning-vintern-1b-v2)
  - [ResNet50 and LSTM Model](#resnet50-and-lstm-model)

## Project Overview
The project develops a VQA system for Vietnamese documents, addressing tasks like extracting information from receipts, signs, or event posters. It integrates multimodal deep learning, processing both visual and textual inputs. Key components include:
- **Image Encoding**: InternViT-300M-448px or ResNet50 extracts visual features.
- **Text Encoding**: PhoBERT encodes Vietnamese questions.
- **Answer Generation**: Transformer decoder, LSTM, or fine-tuned Vintern-1B-v2 generates answers.

Testing scripts validate the fine-tuned Vintern-1B-v2 model on a test dataset, ensuring robust performance in real-world scenarios.

## Dataset

The **ViTextVQA** dataset is used for training and validation, containing images of Vietnamese documents with question-answer pairs. The dataset is available on Hugging Face: [ViTextVQA Dataset](https://huggingface.co/datasets/minhquan6203/ViTextVQA). The dataset includes:

- **Training Set**: `ViTextVQA_train.json` (\~35,159 samples)
- **Validation Set**: `ViTextVQA_dev.json` (\~4,155 samples)

For testing, a separate dataset from Kaggle is used: [Dataset test VQA](https://www.kaggle.com/datasets/zin1610/testdataset-cv).

- **Test Set**: `data_test_cv/My_ViTextVQA_test.json` (\~100 samples).


## Requirements
- Python 3.11+
- PyTorch
- Transformers (Hugging Face, version 4.37.2 for testing)
- EasyOCR
- Underthesea (Vietnamese tokenization)
- Rouge-score, NLTK, pycocoevalcap (evaluation)
- Flash-attn, Timm, Einops, PEFT, Wandb, Deepspeed, Accelerate, Bitsandbytes (fine-tuning and testing)
- PIL, Matplotlib, Pandas, NumPy
- KaggleHub (dataset download)

See `requirements.txt` for a complete list of dependencies.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/5CD-AI/Vintern.git
   cd Vintern
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install timm einops transformers==4.37.2 flash_attn rouge-score pycocoevalcap
   ```

3. **Download Datasets**:

   - Training/Validation:

     ```python
     import kagglehub
     path = kagglehub.dataset_download("tronggg/vitextvqa")
     print(f"Dataset path: {path}")
     ```

   - Test:

     ```python
     path = kagglehub.dataset_download("zin1610/testdataset-cv")
     print(f"Test dataset path: {path}")
     ```

4. **Configure Environment (Optional)**:

   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

**Note**: If `flash_attn` installation fails due to platform issues (as in `Test_my_model.ipynb`), use:

```bash
pip install flash_attn
```

## Usage

### Custom VQA Model

1. **Prepare Data**:

   - Load dataset with `VQADataset` in `vqa-define.ipynb`.
   - Preprocess images using `CLIPImageProcessor` and tokenize text with `underthesea`.

2. **Train**:

   ```python
   model = VQAModel(vocab_size=len(tokenizer), hidden_dim=768).to(device)
   optimizer = AdamW(model.parameters(), lr=1e-4)
   scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
   ```

3. **Inference**:

   ```python
   predicted_ids = model.generate_answer(pixel_values, question_ids, question_mask, bos_id, eos_id)
   predicted_answer = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
   ```

### Fine-tuning Vintern-1B-v2

1. **Set Up**:

   ```bash
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/ComputerVision/CK
   !git clone https://github.com/5CD-AI/Vintern.git
   %cd Vintern
   ```

2. **Install Dependencies**:

   ```bash
   pip install datasets flash_attn timm einops peft wandb deepspeed accelerate bitsandbytes decord tensorboardX gdown
   ```

3. **Fine-tune**:

   - Follow `Fine_tune_my_data.ipynb`.

   - Example:

     ```bash
     python train.py --model_name 5CD-AI/Vintern-1B-v2 --data_path /path/to/ViTextVQA --output_dir /path/to/output
     ```

4. **Inference**:

   ```python
   from transformers import AutoModelForCausalLM, AutoProcessor
   model = AutoModelForCausalLM.from_pretrained("5CD-AI/Vintern-1B-v2").to("cuda")
   processor = AutoProcessor.from_pretrained("5CD-AI/Vintern-1B-v2")
   image = Image.open("path/to/image.jpg")
   question = "Tên cửa hàng là gì?"
   response = model.chat(processor, image, question, generation_config)
   print(response)
   ```

### ResNet50 and LSTM Model

1. **Prepare Data**:

   - Load and preprocess data in `cv-internvit-lstm.ipynb`.
   - Extract features with ResNet50 and tokenize text.

2. **Train**:

   ```python
   model = VQAModelLSTM(vocab_size=len(vocab_answers), hidden_dim=512).to(device)
   optimizer = AdamW(model.parameters(), lr=1e-4)
   ```

3. **Inference**:

   ```python
   predicted_answer = predict_answer(model, image_features, question, vocab_questions, vocab_answers, len_max_question, device)
   print(f"Predicted Answer: {predicted_answer}")
   ```

## Performance
The models were evaluated on the Dataset test VQA in Kaggle using BLEU and ROUGE metrics:

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|------|---------|---------|---------|
| InternViT + LSTM | 0.0036 | 0.1622 | 0.0690 | 0.1311 |
| InternViT + PhoBERT + Fusion-Decoder | 0.0013 | 0.3877 | 0.0466 | 0.2195 |
| Vintern-1B-v2 | 0.1288 | 0.3247 | 0.2380 | 0.3216 |
| Vintern-1B-v2 (Fine-tuned) | 0.4471 | 0.6976 | 0.5735 | 0.6954 |