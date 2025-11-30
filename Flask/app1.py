from flask import Flask, render_template, request, send_file
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from youtube_transcript_api import YouTubeTranscriptApi
import os , re
from dotenv import load_dotenv

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Langchain + Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Flask app
app = Flask(__name__)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Groq Model
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# Extract ID
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else url

# Transcript function (UNCHANGED)
def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    try:
        api = YouTubeTranscriptApi()
        #transcript = api.fetch(video_id)   # keep fetch as you required
        try:
            transcript = api.fetch(video_id, languages=['en'])
        except:
            transcript = api.fetch(video_id, languages=['hi'])

        full_text = " ".join([entry.text for entry in transcript])
        clean_text = re.sub(r'\[.*?\]|\(.*?\)', '', full_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    except Exception as e:
        return f"Transcript Error: {str(e)}"

# Preprocess
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w not in stop_words and w.isalpha()]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    stemmer = PorterStemmer()
    stems = [stemmer.stem(w) for w in lemmas]

    return " ".join(stems)

# Summary
def refine_summarize(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    question_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
You are an expert YouTube video summarizer.
Summarize in this format:
CONTENT:
{text}
"""
    )

    refine_prompt = PromptTemplate(
        input_variables=["existing_answer","text"],
        template="""
We have an existing partial summary.
Refine using new transcript text.

Existing Summary:
{existing_answer}

New Transcript:
{text}

Improve clarity and keep it simple.
"""
    )

    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        verbose=False,
        question_prompt=question_prompt,
        refine_prompt=refine_prompt
    )

    return chain.run(docs)

# Flask home
@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    transcript = None
    clean_text = None

    if request.method == "POST":
        video_url = request.form.get("video_url")

        transcript = get_transcript(video_url)
        if "Error" in transcript:
            return render_template("index.html", error=transcript)

        clean_text = preprocess_text(transcript)
        summary = refine_summarize(transcript)

        # Save summary for download
        with open("summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)

    return render_template(
        "index.html",
        summary=summary,
        transcript=transcript,
        clean_text=clean_text
    )

# Download summary option
@app.route("/download")
def download_summary():
    return send_file("summary.txt", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)