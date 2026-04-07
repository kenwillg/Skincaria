# 🧴 Skincaria — AI Skincare Recommender

An instruction-finetuned GPT-2 model that acts as a **Dermatologist / Beauty Advisor AI**, providing science-backed skincare product recommendations based on user symptoms and skin concerns.

Built as part of the LLM Fine-Tuning (Chapter 7) coursework, following the [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) framework by Sebastian Raschka.

## 📋 Overview

Skincaria bridges the gap between simple product classification and conversational AI. Instead of just labeling skin types, the model **explains why** it recommends specific products — citing active ingredients, their scientific functions, and safety warnings.

### What the model can do:
- **Symptom → Product Recommendation** — "I have oily skin with acne" → recommends specific Sociolla products with ingredient-backed reasoning
- **Product Inquiry** — "Is this product safe for sensitive skin?" → checks ingredient warnings
- **Ingredient Education** — "What is Niacinamide?" → explains function, benefits, and who should avoid it
- **Product Benefits** — "What does this toner do?" → translates Indonesian product claims to English
- **Avoidance Warnings** — "I'm pregnant, what should I avoid?" → flags unsafe ingredients

## 📊 Datasets

This project combines two Kaggle datasets to generate rich, conversational training data:

| Dataset | Source | Description |
|---------|--------|-------------|
| [Indonesian Skincare Sample Dataset](https://www.kaggle.com/datasets/rama87/indonesian-skincare-sample-dataset) | Sociolla | 91 products with ingredients, prices, ratings, and Indonesian product claims |
| [Skin Care Product Ingredients - INCI List](https://www.kaggle.com/datasets/amaboh/skin-care-product-ingredients-inci-list) | renude.co | 248 ingredients with scientific descriptions, skin type suitability, and contraindications |

### Data Processing Pipeline

```
products.xlsx ──────┐
ingredients_cat.xlsx ┼──► generate_dataset.py ──► instruction-data.json (1,563 entries)
product_claims.xlsx ─┤
ingredientsList.csv ─┘
```

The `generate_dataset.py` script:
1. Cross-references product ingredients with INCI scientific profiles
2. Translates Indonesian product claims to English categories
3. Generates 5 types of conversational entries
4. Outputs a Chapter 7-compatible `instruction-data.json`

## 🏗️ Project Structure

```
Skincaria/
├── ch07_skincare.ipynb        # Main training notebook (modified from Ch07)
├── ch07.ipynb                 # Original Chapter 7 notebook (reference)
├── generate_dataset.py        # Data pipeline: CSVs → instruction-data.json
├── instruction-data.json      # Generated training data (1,563 entries)
├── requirements.txt           # Python dependencies
├── dataset/                   # Raw Kaggle datasets
│   ├── Indonesian Skincare Sample Dataset/
│   │   ├── product.xlsx
│   │   ├── ingredients_category.xlsx
│   │   └── product_claim_category.xlsx
│   └── Skin care product ingredients - INCI List/
│       └── ingredientsList.csv
├── .models/                   # Saved model weights (gitignored)
│   └── skincare-recommender-sft.pth
└── gpt2/                      # Pre-trained GPT-2 weights (gitignored)
    └── 355M/
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/kenwillg/Skincaria.git
cd Skincaria

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install llms-from-scratch --no-deps
```

### Generate Training Data

```bash
python generate_dataset.py
```

This outputs `instruction-data.json` with 1,563 conversational entries.

### Train the Model

Open `ch07_skincare.ipynb` in Jupyter or VS Code and run all cells.

> **Note:** Training GPT-2 Medium (355M) requires ~10GB RAM. For local machines with limited memory, use [Google Colab](https://colab.research.google.com) with a free T4 GPU (finishes in ~2 minutes).

## 📈 Training Data Breakdown

| Conversation Type | Count | Description |
|---|---|---|
| Symptom → Recommendation | ~700 | User describes skin concern → model recommends a product with reasoning |
| Product Inquiry | ~150 | User asks if a specific product suits their skin |
| Ingredient Education | ~400 | User learns about a skincare ingredient |
| Product Claims | ~80 | User asks what a product does |
| Avoidance Warnings | ~230 | User asks what ingredients to avoid |
| **Total** | **1,563** | |

## 🔧 Technical Details

- **Base Model:** GPT-2 Medium (355M parameters)
- **Fine-tuning:** Supervised instruction fine-tuning (Alpaca-style prompt template)
- **Framework:** PyTorch (via LLMs from Scratch)
- **Tokenizer:** tiktoken (GPT-2 encoding)
- **Training:** 2 epochs, AdamW optimizer (lr=5e-5, weight_decay=0.1)

## 📝 Example

**Input:**
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Can you recommend a skincare product for me?

### Input:
I have a lot of blackheads on my nose.

### Response:
```

**Output:**
```
For your concern with blackheads, I recommend the KAIE Blemish Care + Pore Refining Serum.
It is a serum that contains Salicylic Acid. Salicylic acid exfoliates the skin's surface,
promoting the removal of dead skin cells, and penetrates deep into pores, helping to unclog
and dissolve sebum and debris. This product is rated 4.8 stars and is priced at Rp 105.950.
```

## 📄 License

This project is for educational purposes as part of university coursework.

## 🙏 Acknowledgments

- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka
- [Indonesian Skincare Sample Dataset](https://www.kaggle.com/datasets/rama87/indonesian-skincare-sample-dataset) by Rama87
- [Skin Care Product Ingredients - INCI List](https://www.kaggle.com/datasets/amaboh/skin-care-product-ingredients-inci-list) by amaboh
