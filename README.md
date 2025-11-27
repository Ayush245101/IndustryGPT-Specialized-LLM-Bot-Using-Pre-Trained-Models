# IndustryGPT – Transformer-Based E‑Commerce Support Chatbot

IndustryGPT is a **domain-specific chatbot** for e‑commerce customer support, built by **fine‑tuning a T5‑small transformer** on a curated dataset of realistic customer–agent conversations.  
It is packaged as a simple **web app using Streamlit**, and can be shared via **ngrok** for quick demos and testing.

---

## ✨ Features

- 🤖 **Domain‑specific chatbot** for common e‑commerce queries:
  - Order tracking, cancellations, returns, payments, invoices, account help, etc.
- 🧠 **Transformer-based model**:
  - Fine‑tuned [`t5-small`](https://huggingface.co/t5-small) for query → response generation.
- 📚 **Custom dataset**:
  - 28k+ synthetic but realistic customer–agent dialogue pairs.
- 🌐 **Web UI with Streamlit**:
  - Simple chat interface you can run locally.
- 🌍 **Ngrok integration (optional)**:
  - Expose your local app via a public URL for demos.

---

## 🏗️ Project Architecture

High-level flow of the system:

```text
User (browser)
    │
    ▼
Streamlit Web App  ──►  IndustryGPT (fine‑tuned T5‑small)
    │                          │
    └─────────── Response ◄────┘
```

Optionally, **ngrok** tunnels the local Streamlit server to a public URL for sharing.

---

## 📂 Repository Structure

> Note: File names may vary slightly depending on your setup – adjust this section to match your repo.

```text
.
├─ data/
│  └─ data.csv              # Customer query → agent response pairs
├─ notebooks/
│  └─ training.ipynb        # Colab / Jupyter notebook used to fine‑tune T5‑small
├─ model/
│  └─ finetuned-t5-bot/     # Saved model + tokenizer (optional, or downloaded at runtime)
├─ app.py                   # Streamlit chat app for IndustryGPT
├─ requirements.txt         # Python dependencies
└─ README.md                # You are here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/IndustryGPT-Specialized-LLM-Bot-Using-Pre-Trained-Models.git
cd IndustryGPT-Specialized-LLM-Bot-Using-Pre-Trained-Models
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies (if you need to rebuild `requirements.txt`):

```text
transformers
torch
pandas
numpy
streamlit
pyngrok
datasets
```

---

## 🧠 Training the Model (Optional)

If the repository already contains a `model/finetuned-t5-bot` directory, you can **skip training** and go straight to running the app.

Otherwise, to re‑train or fine‑tune:

1. Open the notebook in `notebooks/training.ipynb` (or your Colab link).
2. Make sure the dataset CSV path is correct (e.g., `data/data.csv`).
3. Run all cells to:
   - Load and preprocess the dataset.
   - Load `t5-small`.
   - Fine‑tune on the e‑commerce dialogue pairs.
   - Save the model and tokenizer to `./model/finetuned-t5-bot`.

Adjust the number of epochs, batch size, and learning rate based on your hardware.

---

## 💬 Running the Chatbot (Local)

Once the fine‑tuned model is available in `./model/finetuned-t5-bot`:

```bash
streamlit run app.py
```

Then open the URL shown in your terminal, usually:

```text
http://localhost:8501
```

You should see:

- A title for the chatbot.
- A text input box to type your question.
- A chat history area showing your messages and bot responses.

---

## 🌐 Sharing via Ngrok (Optional)

To let others access your chatbot over the internet:

1. Install ngrok and pyngrok:

   ```bash
   pip install pyngrok
   ```

2. Get a free ngrok auth token from:  
   [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

3. In your code (or a small helper script):

   ```python
   from pyngrok import ngrok

   ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN_HERE")
   public_url = ngrok.connect(8501)  # Streamlit default port
   print("Public URL:", public_url)
   ```

4. Start Streamlit in a terminal:

   ```bash
   streamlit run app.py
   ```

5. Share the **public URL** printed by ngrok with your teammates or supervisor.

---

## 🧩 How It Works (Concept Summary)

1. **Dataset**  
   - CSV file with two main columns:
     - `Human_Query` – customer message.
     - `Assistant_Response` – desired, helpful reply.
   - Covers common e‑commerce support topics (orders, returns, billing, etc.).

2. **Model**  
   - Pre‑trained [`t5-small`](https://huggingface.co/t5-small) from Hugging Face.
   - Fine‑tuned to map `Human_Query` → `Assistant_Response`.

3. **Inference**  
   - User input is formatted into a prompt (e.g., `User: <text>\nBot:`).
   - Model generates the continuation, which forms the bot reply.

4. **Web App**  
   - Streamlit handles the UI and calls the model on each user message.
   - Session state is used to keep a simple chat history.

---

## 📊 Limitations

- The dataset is **synthetic/template‑based**; not all real-world edge cases are covered.
- The prototype **does not integrate** with live order/billing systems (no real account lookups).
- Evaluation is mostly **qualitative** (by looking at answers), not a full user study.

---

## 🛠️ Possible Improvements

If you want to extend this project:

- Add **real anonymized logs** (if available and compliant with privacy rules).
- Integrate a **database or API** for live order/account queries.
- Add **retrieval‑augmented generation (RAG)** to ground answers in a knowledge base.
- Add **safety filters** and escalation to human agents for complex/sensitive cases.
- Build a **multi-language** version with additional training data.

---

## 📚 References & Related Work

A few useful references on transformer‑based chatbots for customer support:

- Customer service chatbot enhancement with attention-based transfer learning  
  [https://www.sciencedirect.com/science/article/pii/S0950705124009274](https://www.sciencedirect.com/science/article/pii/S0950705124009274)

- Conversational AI For E‑Commerce Customer Service  
  [https://ijcrt.org/papers/IJCRT2502172.pdf](https://ijcrt.org/papers/IJCRT2502172.pdf)

- Enhancing Customer Support Chatbots with LLMs: Comparative Analysis of Few-Shot Learning, Fine-Tuning, and RAG  
  [https://studenttheses.uu.nl/bitstream/handle/20.500.12932/48393/Master_thesis_Rowan_Woering_6570941.pdf?sequence=1](https://studenttheses.uu.nl/bitstream/handle/20.500.12932/48393/Master_thesis_Rowan_Woering_6570941.pdf?sequence=1)

- A systematic literature review on AI chatbots in automating customer support for e‑commerce  
  [https://iacis.org/iis/2025/1_iis_2025_403-417.pdf](https://iacis.org/iis/2025/1_iis_2025_403-417.pdf)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add my feature"`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Open a Pull Request.

---

## 📄 License

Add your chosen license here (e.g., MIT, Apache 2.0).

```text
This project is licensed under the MIT License – see the LICENSE file for details.
```

---

## 🧑‍💻 Author

- **Ayush** – initial work and implementation.  
  GitHub: [https://github.com/Ayush245101](https://github.com/Ayush245101)
