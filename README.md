# Dead Reckoning — Spec Inference Engine for Undocumented Industrial Parts

An AI system that reconstructs complete product specifications for industrial parts where **no external data exists** — using engineering domain reasoning, physics-based inference, and vision analysis.

Unlike existing tools that scrape or lookup data, Dead Reckoning **derives** specs from first principles.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Yash151005/Dead-Reckoning.git
cd Dead-Reckoning
pip install -r requirements.txt
```

### 2. Get a Groq API Key

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up / log in
3. Navigate to **API Keys** and create a new key
4. Copy the key — you'll paste it into the app sidebar

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Enter your Groq API key in the sidebar to get started.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

No secrets configuration needed — users enter their own Groq API key in the sidebar at runtime.

---

## 🧠 How It Works

| Step | Description |
|------|-------------|
| **1. Input Fragments** | User provides partial part info, context, and optional photo |
| **2. Vision Analysis** | AI reads nameplates, markings, and part geometry from images |
| **3. Domain Reasoning** | LLM applies 30+ years of engineering knowledge |
| **4. Spec Derivation** | Physics-based inference reconstructs full spec sheet |
| **5. Confidence Scoring** | Each field rated by derivation method and certainty |

---

## 📋 Features

- **Spec Inference** — Reconstruct full specs from fragments + context
- **Nameplate Decoder** — Extract structured data from worn/damaged nameplates
- **Batch Inference** — Process multiple parts via CSV upload
- **Confidence Tracking** — Every field tagged as MEASURED / DERIVED / INFERRED / ASSUMED
- **Export** — Download results as JSON or CSV

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM (text) | Groq — `openai/gpt-oss-120b` |
| LLM (vision) | Groq — `qwen/qwen3.6-27b` |
| Image processing | Pillow |
| Data handling | Pandas |

---

## 📁 File Structure

```
dead-reckoning/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore file
└── README.md           # This file
```

---

## 📄 License

MIT License — built for **UniHack 2026**.

---

> **Dead Reckoning** — *When the datasheet is gone, engineering reasoning finds the way.*
