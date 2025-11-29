# 📦 IndustryGPT – E-Commerce Customer Service LLM (LoRA-Fine-Tuned T5)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/WebUI-Streamlit-brightgreen.svg)](https://streamlit.io/)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)

**IndustryGPT** is a domain-specialized **E-Commerce Customer Support Chatbot** built by fine-tuning a **T5-small** model using **LoRA (Low-Rank Adaptation)**.  
It handles customer queries like **order tracking, returns, refunds, payments, delivery issues, cancellations**, and more.

The bot is deployed via a simple, clean **Streamlit** chat interface and can be tested in **Google Colab** or run locally.

> 🔗 Repo: **IndustryGPT-Specialized-LLM-Bot-Using-Pre-Trained-Models**  
> https://github.com/Ayush245101/IndustryGPT-Specialized-LLM-Bot-Using-Pre-Trained-Models

---

## 🚀 Features

- 🎯 **Domain-specific** E-Commerce customer support chatbot  
- ⚡ **LoRA fine-tuning** on `t5-small` → parameter-efficient training  
- 🧹 Robust **text preprocessing** (lowercasing, cleaning, preserving placeholders)  
- 🧠 Training with **HuggingFace `Trainer`** + `accelerate`  
- 💬 **Interactive CLI chat loop** for quick testing  
- 🌐 **Streamlit web app** (+ optional ngrok for public URL)  
- 🧱 **Modular design** → easy to swap datasets, models, domains

---

## 🧩 Tech Stack

- **Language Model:** `t5-small` (HuggingFace Transformers)  
- **Fine-Tuning:** LoRA via [`peft`](https://github.com/huggingface/peft)  
- **Training:** `transformers.Trainer`, `accelerate`, `bitsandbytes` (optional)  
- **Data Handling:** `pandas`, `datasets`  
- **Serving / UI:** `streamlit`, `pyngrok`  
- **Runtime:** Python 3.x, GPU (T4 on Colab recommended)

---

## 📂 Project Structure

```text
.
├── IndustryGPT_–_E-Commerce_Customer_Service_LLM_with_LoRA.ipynb  # Main Colab / Jupyter notebook
├── app.py                                                        # Streamlit web app
├── finetuned-t5-lora-bot/                                        # Saved LoRA-adapted model + tokenizer
├── data (1).csv                                                  # Training dataset (Human_Query / Assistant_Response)
└── README.md                                                     # Project documentation

---

## 📊 Dataset

- Input column: `Human_Query` → renamed to `human_query`
- Target column: `Assistant_Response` → renamed to `assistant_response`

### Preprocessing

- Convert to lowercase.
- Strip leading/trailing whitespace.
- Collapse multiple spaces.
- Preserve placeholders like `{{order number}}`.
- Drop rows with missing text.

A cleaned version is stored as:

- `human_query_clean`
- `assistant_response_clean`

The model is trained on:

```text
user: <human_query_clean>
bot:
```

with the target being `assistant_response_clean`.

---

## 🧠 Model & LoRA Setup

- Base model: `t5-small`
- Task type: `SEQ_2_SEQ_LM`
- LoRA config:
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0.1`
  - `target_modules = ["q", "v"]` (attention projections)

Only the LoRA adapter parameters are trained; the rest of the model remains frozen, making training efficient even on a single T4 GPU.

---

## 🏋️ Training Workflow (Notebook)

All steps are implemented in  
`IndustryGPT_–_E‑Commerce_Customer_Service_LLM_with_LoRA.ipynb`.

High‑level steps:

1. **Install dependencies**

   ```bash
   pip install -q transformers datasets sentencepiece peft accelerate bitsandbytes
   ```

2. **Load and preprocess data**

   - Read `data (1).csv`
   - Rename columns, clean text
   - Train/test split using `datasets.DatasetDict`

3. **Load base T5 + tokenizer**

   ```python
   MODEL_NAME = "t5-small"
   tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
   base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
   ```

4. **Wrap model with LoRA**

   ```python
   lora_config = LoraConfig(
       task_type=TaskType.SEQ_2_SEQ_LM,
       r=16,
       lora_alpha=32,
       lora_dropout=0.1,
       target_modules=["q", "v"],
   )
   model = get_peft_model(base_model, lora_config)
   ```

5. **Tokenization for training**

   Format:

   ```text
   user: <cleaned customer query>
   bot:
   ```

   with max lengths `128` for inputs and targets.

6. **Training**

   - Uses `transformers.Trainer`
   - 5 epochs, `learning_rate = 1e-4`, batch size `8`
   - Logs training loss; evaluation loss is available if `evaluation_strategy` is enabled.

7. **Save model**

   ```python
   save_dir = "./finetuned-t5-lora-bot"
   trainer.save_model(save_dir)
   tokenizer.save_pretrained(save_dir)
   ```

---

## 💬 Inference (Chat Bot)

In the notebook, an inference helper loads base T5 + LoRA adapters and defines:

```python
def chat_with_bot(user_input: str, max_new_tokens: int = 64) -> str:
    prompt = f"user: {user_input.strip().lower()}\nbot:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = lora_model_infer.generate(
            **inputs,
            max_new_tokens=500,
            min_new_tokens=20,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()
```

You can also run a simple CLI loop from the notebook:

```python
def chat_loop():
    print("🟢 LoRA T5 ChatBot is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break
        reply = chat_with_bot(user_input)
        print(f"Bot: {reply}\n")

# chat_loop()
```

---

## 🌐 Streamlit Web App

The notebook generates an `app.py` that:

- Loads the fine‑tuned LoRA model from `./finetuned-t5-lora-bot`
- Opens a chat‑style UI with history
- Handles inference using the same prompt format

### Running the Web App (e.g., in Colab)

```bash
pip install streamlit pyngrok
```

In Python (e.g., notebook cell):

```python
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")
!streamlit run app.py &>/content/logs.txt &

public_url = ngrok.connect(8501)
print("Public URL:", public_url)
```

Open the printed ngrok URL in your browser to chat with IndustryGPT.

---

## 🎯 Goals & Objectives

**High‑Level Goal**

Build a **scalable, domain-specific AI chatbot** for e‑commerce customer support that can be trained and deployed efficiently using **LoRA parameter‑efficient fine‑tuning**.

**Specific Objectives**

1. **Data Preparation & Preprocessing**
   - Clean and normalize customer queries and responses.
   - Preserve important templates (order IDs, placeholders).
   - Create a robust train/test split for evaluation.

2. **Model Adaptation with LoRA**
   - Start from a general-purpose T5 model.
   - Apply LoRA to selected attention layers.
   - Reduce trainable parameters and memory usage.

3. **Training & Evaluation**
   - Use Hugging Face `Trainer` for reproducible training.
   - Monitor training loss and (optionally) evaluation loss.
   - Experiment with hyperparameters (epochs, LR, LoRA rank).

4. **Interactive Inference**
   - Provide a Python function (`chat_with_bot`) for quick testing.
   - Develop a Streamlit app to simulate a customer support widget.

5. **Extensibility**
   - Keep the pipeline modular:
     - Swap datasets.
     - Add new domains or languages using new LoRA adapters.
     - Extend to include retrieval, intent classification, or backend integrations.

---

## 🛠️ Setup & Usage (Local / Non‑Colab)

1. **Clone the repo**

   ```bash
   git clone https://github.com/Ayush245101/IndustryGPT-Specialized-LLM-Bot-Using-Pre-Trained-Models.git
   cd IndustryGPT-Specialized-LLM-Bot-Using-Pre-Trained-Models
   ```

2. **Create and activate a virtual env (optional but recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt   # if you add one
   # or manually:
   pip install transformers datasets sentencepiece peft accelerate bitsandbytes streamlit pyngrok pandas torch
   ```

4. **Run training**

   - Open the notebook `IndustryGPT_–_E‑Commerce_Customer_Service_LLM_with_LoRA.ipynb` in Jupyter / VSCode / Colab.
   - Run cells sequentially to train and save the model.

5. **Run Streamlit app locally**

   After training (and saving `finetuned-t5-lora-bot`):

   ```bash
   streamlit run app.py
   ```

   Then open `http://localhost:8501` in your browser.

---

## 📌 What’s Implemented vs. Possible Improvements

**Implemented**

- Full LoRA training pipeline on `t5-small`.
- Clean preprocessing and prompt formatting.
- CLI chat loop.
- Streamlit web UI + optional ngrok tunneling.
- Project description, goals, and summary in notebook.

**Potential Improvements**

- Add proper **evaluation metrics** (BLEU, ROUGE, etc.).
- Implement **logging & experiment tracking** (Weights & Biases).
- Add **intent routing** (different flows for returns, cancellations, payments).
- Integrate with a **real order database** or mock backend.
- Support **multilingual** customer support.

---

## 🤝 Contributing

Contributions and suggestions are welcome!

- Fork the repo
- Create a feature branch
- Open a Pull Request with a clear description

---

## 📄 License

Add your chosen license here (e.g., MIT, Apache 2.0). Example:

```text
This project is licensed under the MIT License.
```

---

## 🙋 Contact

- Author: **Ayush**
- GitHub: [Ayush245101](https://github.com/Ayush245101)

If you’d like, I can also generate a minimal `requirements.txt` matching this setup.
