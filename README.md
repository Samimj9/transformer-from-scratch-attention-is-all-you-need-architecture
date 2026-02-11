

# Transformer Architecture from Scratch (PyTorch)

A modular, from-scratch implementation of the original Transformer model as described in the seminal paper **"Attention Is All You Need"**. This project implements the full Encoder-Decoder stack and was validated by training a translation model from English to German.

🏗️ Architecture Reference
![Transformer Architecture](<img width="471" height="446" alt="image" src="https://github.com/user-attachments/assets/9f042e36-fc33-4bce-8f51-4637cfbba7d3" />)


> Citation:Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*.

 📂 Project Structure

The repository is organized into modular packages :

* model/: Contains the core architecture components.
* attention.py: Multi-Head Attention mechanism.
* layers.py: Encoder/Decoder layers and Feed-Forward networks.
* embeddings.py: Sinusoidal Positional Encodings.
* transformer.py: The full Transformer assembly.


* utils/: Data processing logic.
* data_utils.py: Custom Vocabulary class and tokenization pipeline.


* main.py: Orchestrates the training process.
* translate.py: Inference script for testing new sentences.

## 🚀 Key Features

* Manual Attention Implementation: Built the Scaled Dot-Product and Multi-Head Attention from scratch.
* Causal Masking: Implemented look-ahead masks to ensure the decoder predicts tokens auto-regressively.
* Custom Vocabulary: Built a dedicated mapping for English and German tokens using Hugging Face `datasets`.
* Teacher Forcing: Implemented during training for stable and faster convergence.

📊 Results

The model was trained on the Multi30k dataset. Below is a real inference result demonstrating the model's ability to capture both meaning and grammar:

English Input: > "A dog runs in the park."

German Output:> "<bos> ein hund rennt im park durch den park . <eos>"

🛠️ Setup and Installation

1. Install dependencies:**

pip install -r requirements.txt




2. Download Spacy Language Models:**

python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm




3. Train the model:

python main.py




4. Test translation:

python translate.py





