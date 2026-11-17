# 🎓 Student Support FAQ Chatbot

## 📌 Project Overview

The Student Support FAQ Chatbot is an AI-powered chatbot developed using Python and Streamlit. It helps students find answers to frequently asked university-related questions and provides responses to general educational and technology questions.

The chatbot uses Natural Language Processing (NLP) techniques to identify relevant questions from an FAQ dataset. If a suitable FAQ match is not found, it uses Google's Gemini AI to generate a response.

## ✨ Features

- Interactive chatbot interface.
- FAQ-based question answering.
- Natural Language Processing using NLTK.
- TF-IDF text vectorization.
- Cosine Similarity for question matching.
- Gemini AI integration for general questions.
- Chat history management.
- Clear chat functionality.
- User-friendly interface with a custom design.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NLTK
- Scikit-learn
- Google Gemini API
- Python-dotenv

## 📂 Project Structure

```text
CodeAlpha_FAQChatbot/
│
├── app.py
├── faq_data.csv
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd CodeAlpha_FAQChatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Packages

```bash
pip install -r requirements.txt
```

### 6. Configure Gemini API Key

Create a `.env` file in the project folder and add:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your own Gemini API key.

### 7. Run the Application

```bash
streamlit run app.py
```

The chatbot will open in your web browser.

## 🤖 How It Works

1. The user enters a question through the chatbot interface.
2. The application preprocesses the question using NLTK.
3. TF-IDF converts the question into numerical vectors.
4. Cosine Similarity compares the user's question with questions in the FAQ dataset.
5. If a suitable match is found, the chatbot displays the corresponding FAQ answer.
6. Otherwise, the Gemini AI API generates a response.

## 👩‍💻 Developer

Developed as part of the CodeAlpha Python Programming Internship.

## 🌐 Live Demo

Try the Student Support FAQ Chatbot:

👉 **[Launch Student Support Chatbot](YOUR_STREAMLIT_APP_URL)**

Experience AI-powered student support with FAQ matching and Gemini AI integration.