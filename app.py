import streamlit as st
import pandas as pd
import nltk
import re
import os
import html

from dotenv import load_dotenv
from google import genai

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Support Chatbot",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# 2. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"


# ==========================================
# 3. INITIALIZE GEMINI CLIENT
# ==========================================

@st.cache_resource
def get_gemini_client():

    if not GEMINI_API_KEY:
        return None

    return genai.Client(api_key=GEMINI_API_KEY)


client = get_gemini_client()


# ==========================================
# 4. DOWNLOAD NLTK RESOURCES
# ==========================================

@st.cache_resource
def download_resources():

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    return set(stopwords.words("english"))


stop_words = download_resources()


# ==========================================
# 5. CUSTOM CSS DESIGN
# ==========================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #F4F5FF;
}

/* Main Title */
.main-title {
    text-align: center;
    color: #302064 !important;
    font-size: 36px;
    font-weight: 800;
    margin-block-end: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #333333 !important;
    font-size: 17px;
    margin-block-end: 30px;
}

/* General Headings */
h1, h2, h3, h4 {
    color: #302064 !important;
    font-weight: 700 !important;
}

/* General Text */
.stApp p,
.stApp li,
.stApp label {
    color: #202020 !important;
}

/* Chat Message Containers */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF;
    border: 1px solid #D5D0EA;
    border-radius: 16px;
    padding: 12px 16px;
    margin-block-end: 14px;
}

/* Chat Message Text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #171717 !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
}

/* Chat Message Headings */
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] b {
    color: #302064 !important;
    font-weight: 750 !important;
}


/* ==========================================
   ADVANCED CHAT INPUT DESIGN
========================================== */

/* Chat Input Container */
[data-testid="stChatInput"] {
    background-color: transparent !important;
    padding: 10px 0 !important;
}

/* Input Box */
[data-testid="stChatInput"] > div {
    background: #EAE7FF !important;
    border: 2px solid #C9BFFF !important;
    border-radius: 22px !important;
    padding: 7px 12px !important;

    box-shadow: 0 5px 18px rgba(73, 54, 168, 0.10) !important;

    transition: all 0.3s ease !important;
}

/* Input Focus Effect */
[data-testid="stChatInput"] > div:focus-within {
    background: #F0EDFF !important;
    border: 2px solid #7558D8 !important;

    box-shadow: 0 5px 22px rgba(73, 54, 168, 0.20) !important;
}

/* Input Text */
[data-testid="stChatInput"] textarea {
    color: #211747 !important;
    background-color: transparent !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    caret-color: #4936A8 !important;
}

/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #625A83 !important;
    opacity: 1 !important;
}

/* Send Button */
[data-testid="stChatInput"] button {
    background-color: #4936A8 !important;
    border-radius: 50% !important;
    border: none !important;
}

/* Send Button Hover */
[data-testid="stChatInput"] button:hover {
    background-color: #302064 !important;
}


/* ==========================================
   SIDEBAR DESIGN
========================================== */

[data-testid="stSidebar"] {
    background-color: #EAE7FF;
}

/* Sidebar Headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #302064 !important;
}

/* Sidebar Text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #252044 !important;
    font-size: 15px !important;
}

/* Buttons */
.stButton button {
    background-color: #4936A8;
    color: #FFFFFF !important;
    border-radius: 10px;
    font-weight: 600;
    border: none;
}

/* Button Hover */
.stButton button:hover {
    background-color: #352477;
    color: #FFFFFF !important;
}

/* Links */
.stApp a {
    color: #4936A8 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 6. TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = word_tokenize(text)

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


# ==========================================
# 7. LOAD FAQ DATASET
# ==========================================

# ==========================================
# 7. LOAD FAQ DATASET WITH ERROR HANDLING
# ==========================================

@st.cache_data
def load_faq():

    try:
        data = pd.read_csv("faq_data.csv")

        if data.empty:
            raise ValueError("FAQ dataset is empty.")

        required_columns = ["question", "answer"]

        for column in required_columns:
            if column not in data.columns:
                raise ValueError(
                    f"Missing required column: {column}"
                )

        data = data.dropna(subset=["question", "answer"])

        if data.empty:
            raise ValueError("No valid FAQ records found.")

        return data

    except FileNotFoundError:
        st.error("FAQ dataset file not found. Please check faq_data.csv.")
        st.stop()

    except Exception as e:
        st.error(f"Error loading FAQ dataset: {e}")
        st.stop()


faq_data = load_faq()


# ==========================================
# 8. TF-IDF VECTORIZATION
# ==========================================

@st.cache_resource
def create_vectors(questions):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(questions)

    return vectorizer, vectors


faq_data["processed_question"] = (
    faq_data["question"].apply(preprocess_text)
)

vectorizer, faq_vectors = create_vectors(
    faq_data["processed_question"].tolist()
)


# ==========================================
# 9. GEMINI AI RESPONSE
# ==========================================

def get_gemini_response(user_question):

    if client is None:

        return (
            "Gemini API key was not found. "
            "Please check your .env file."
        )

    try:

        prompt = f"""
You are a friendly and helpful Student Support AI Chatbot.

Your role is to answer student questions
in clear and simple English.

Instructions:

1. Give accurate and helpful answers.
2. Keep answers clear and easy to understand.
3. If the question is about university-specific information
   that you do not know, do not invent details.
4. Politely suggest contacting the relevant university
   department when exact information is required.
5. Answer general educational and technology questions as well.
6. Use simple examples when helpful.

Student Question: {user_question}
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if response.text:
            return response.text

        return "Sorry! I couldn't generate an answer. Please try again."

    except Exception as e:

        print("Gemini API Error:", e)

        return (
            "Sorry! The AI service is temporarily unavailable. "
            "Please try again later."
        )


# ==========================================
# 10. MAIN CHATBOT RESPONSE
# ==========================================

def get_answer(user_question):

    processed_question = preprocess_text(user_question)

    if not processed_question.strip():

        return "Please enter a meaningful question."

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_index = similarity.argmax()

    best_score = similarity[0][best_index]

    # FAQ Matching
    if best_score >= 0.20:

        return faq_data.iloc[best_index]["answer"]

    # Gemini AI Fallback
    return get_gemini_response(user_question)


# ==========================================
# 11. CHATBOT HEADER
# ==========================================

st.markdown(
    """
    <div class="main-title">
        🎓 Student Support Chatbot
    </div>

    <div class="subtitle">
        Your virtual assistant for student inquiries
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 12. INITIALIZE CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# 13. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("ℹ️ About")

    st.write(
        "This chatbot uses Natural Language Processing "
        "and Gemini AI to answer student questions."
    )

    st.subheader("✨ Features")

    st.write("• FAQ-based responses")
    st.write("• AI-powered answers")
    st.write("• Natural Language Processing")
    st.write("• Instant student support")

    st.subheader("💡 Sample Questions")

    sample_questions = [

        "How can I apply for admission?",

        "Where is the campus located?",

        "How can I reset my password?",

        "Is Wi-Fi available on campus?",

        "How can I pay course fees?",

        "What is Artificial Intelligence?",

        "Explain Python programming."

    ]

    for question in sample_questions:

        st.write("•", question)

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==========================================
# 14. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user", avatar="👤"):

            st.markdown(
                html.escape(message["content"])
            )

    else:

        with st.chat_message("assistant", avatar="🤖"):

            st.markdown(
                message["content"]
            )


# ==========================================
# 15. CHAT INPUT
# ==========================================

user_input = st.chat_input(
    "Type your question here..."
)


if user_input:

    # Add user message
    st.session_state.messages.append({

        "role": "user",

        "content": user_input

    })

    # Generate chatbot response
    with st.spinner("🤖 Thinking..."):

        answer = get_answer(user_input)

    # Add chatbot response
    st.session_state.messages.append({

        "role": "assistant",

        "content": answer

    })

    # Refresh application
    st.rerun()