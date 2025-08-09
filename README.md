# TinyStories SLM: A PyTorch Implementation 
This project is a from-scratch implementation of a Small Language Model (SLM) in PyTorch, based on the concepts presented in the research paper [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).

The goal is to build a decoder-only Transformer with approximately 30 million parameters that can generate creative and coherent short stories after being trained on the TinyStories dataset.

## Project Goal & Motivation 
The original "TinyStories" paper demonstrates that language models don't need to be enormous to generate fluent, logical text. By training on a carefully curated, high-quality dataset of simple stories, even a relatively small model can learn grammar, reasoning, and storytelling.

This repository aims to replicate this fascinating finding by building and training a similar small-scale model from the ground up.

## Features 
Framework: Built entirely in PyTorch.

Architecture: GPT-style decoder-only Transformer, inspired by the paper's models.

Tokenization: Uses tiktoken with the gpt2 vocabulary.

Modern Training Techniques:

AdamW optimizer for stable training.

Cosine annealing learning rate scheduler with a linear warmup.

Mixed-precision training (bfloat16/float16) for efficiency.

Gradient accumulation to simulate larger batch sizes.

Gradient clipping to prevent exploding gradients.

## How to Run 🚀
1. Clone the Repository

Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install Dependencies

Create a requirements.txt file with the following content:

torch
datasets
tiktoken
numpy
tqdm
matplotlib
Then, install the packages:

Bash
pip install -r requirements.txt
3. Run the Training Script

Execute the main Python file. The script will automatically download the dataset, tokenize it, and begin training.

Bash
python your_script_name.py
The script saves the best-performing model to best_model_params.pt and plots the training/validation loss curves in loss_curves.png.

## Model Architecture ⚙️
The model's architecture is designed to be similar in scale to one of the models discussed in the paper. The following hyperparameters result in a model with approximately 30 million trainable parameters:

Vocabulary Size: 50,257

Embedding Dimension (n_embd): 384

Number of Layers (n_layer): 6

Number of Attention Heads (n_head): 6

Context Window (block_size): 128

Dropout Rate: 0.1

## Output
After training, the script will:

Save the best model weights to best_model_params.pt.

Generate a loss_curves.png image showing training and validation loss.

Run inference on two sample prompts and print the generated stories to the console.
