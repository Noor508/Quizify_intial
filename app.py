import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Define the MCQ and List_of_MCQs data models
class MCQ(BaseModel):
    question: str
    options: List[str]
    answer: str

class List_of_MCQs(BaseModel):
    mcqs: List[MCQ]

# Initialize session state
if "current_text" not in st.session_state:
    st.session_state.current_text = ""
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
if "all_mcqs" not in st.session_state:
    st.session_state.all_mcqs = []

# Initialize OpenAI API
st.session_state.bot = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def split_text(text, max_length):
    """Split text into chunks of a maximum length."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def generate_mcqs_from_text(text):
    """Generate MCQs from text using OpenAI."""
    example_json = [
        {
            "question": "What is the capital of France?",
            "options": ["Paris", "London", "Berlin", "Madrid"],
            "answer": "Paris"
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "options": ["Earth", "Mars", "Jupiter", "Saturn"],
            "answer": "Mars"
        }
    ]
    
    text_chunks = split_text(text, 1000)  # Adjust chunk size as needed
    all_mcqs = []

    for chunk in text_chunks:
        prompt = f"Generate multiple-choice questions (MCQs) from the following text in JSON format:\n\n{chunk}"
        messages = [
            {"role": "system", "content": f"Generate MCQs from the following text in JSON format:\n{json.dumps(example_json, indent=4)}"},
            {"role": "user", "content": chunk}
        ]
        response = st.session_state.bot.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        mcqs = json.loads(response.choices[0].message['content'])
        all_mcqs.extend(mcqs)
    
    return all_mcqs

def get_transcript(youtube_url):
    """Extract transcript from a YouTube video."""
    video_id = extract_video_id(youtube_url)
    if video_id:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
            return transcript_text
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")
            return ""
    else:
        st.error("Invalid YouTube URL.")
        return ""

def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    if "youtube.com/watch?v=" in url:
        return url.split("youtube.com/watch?v=")[-1]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1]
    return None

st.title("PDF and YouTube Quiz Generator")

tab1, tab2 = st.tabs(["PDF to MCQs", "YouTube Link Quiz"])

with tab1:
    st.header("PDF to MCQs")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_reader = PyPDFLoader("uploaded_file.pdf")
        documents = pdf_reader.load()
        pdf_text = "\n".join([doc.page_content for doc in documents])

        os.remove("uploaded_file.pdf")

        st.session_state.current_text = pdf_text
        st.session_state.all_mcqs = generate_mcqs_from_text(pdf_text)
        
    if st.session_state.all_mcqs:
        all_mcqs = st.session_state.all_mcqs
        current_question = st.session_state.current_question

        if current_question < len(all_mcqs):
            mcq = all_mcqs[current_question]

            st.write(f"**Question {current_question + 1}:** {mcq['question']}")
            selected_option = st.radio("Select an option:", mcq['options'], key=f"question_{current_question}")

            if st.button("Submit"):
                st.session_state.selected_option = selected_option
                st.session_state.show_feedback = True

            if st.session_state.show_feedback:
                if st.session_state.selected_option == mcq['answer']:
                    st.success("Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"Wrong! The correct answer is: {mcq['answer']}")

                if st.button("Next"):
                    st.session_state.current_question += 1
                    st.session_state.show_feedback = False
                    st.session_state.selected_option = None
                    st.rerun()
        else:
            st.write(f"Quiz completed! Your score is {st.session_state.score} out of {len(all_mcqs)}.")
    else:
        st.write("Please upload a PDF file to generate MCQs.")

with tab2:
    st.header("YouTube Link Quiz")
    youtube_url = st.text_input("Enter YouTube video URL:")

    if youtube_url:
        transcript_text = get_transcript(youtube_url)
        
        if transcript_text:
            st.session_state.current_text = transcript_text
            st.session_state.all_mcqs = generate_mcqs_from_text(transcript_text)
            
        if st.session_state.all_mcqs:
            all_mcqs = st.session_state.all_mcqs
            current_question = st.session_state.current_question

            if current_question < len(all_mcqs):
                mcq = all_mcqs[current_question]

                st.write(f"**Question {current_question + 1}:** {mcq['question']}")
                selected_option = st.radio("Select an option:", mcq['options'], key=f"question_{current_question}")

                if st.button("Submit"):
                    st.session_state.selected_option = selected_option
                    st.session_state.show_feedback = True

                if st.session_state.show_feedback:
                    if st.session_state.selected_option == mcq['answer']:
                        st.success("Correct!")
                        st.session_state.score += 1
                    else:
                        st.error(f"Wrong! The correct answer is: {mcq['answer']}")

                    if st.button("Next"):
                        st.session_state.current_question += 1
                        st.session_state.show_feedback = False
                        st.session_state.selected_option = None
                        st.rerun()
            else:
                st.write(f"Quiz completed! Your score is {st.session_state.score} out of {len(all_mcqs)}.")
        else:
            st.write("Please enter a valid YouTube URL to generate a quiz.")
