import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.environ.get("API_KEY")
endpoint = os.environ.get("END_POINT")

headers = {
    'Content-Type': 'application/json',
    'api-key': api_key
}

# dummy_mcqs = [{'question': 'What type of data does supervised learning use?', 'options': ['Unlabeled data', 'Labeled data', 'Random data', 'Synthetic data'], 'answer': 'Labeled data'}, {'question': 'Which subcategory can supervised learning be divided into?', 'options': ['Reinforcement learning', 'Clustering', 'Classification', 'Dimensionality reduction'], 'answer': 'Classification'}, {'question': 'Which of the following is an example of a classification algorithm?', 'options': ['Support Vector Machines', 'Linear Regression', 'Logistic Regression', 'Clustering'], 'answer': 'Support Vector Machines'}, {'question': 'What is a common application of clustering in unsupervised learning?', 'options': ['Predicting house prices', 'Customer segmentation', 'Spam detection', 'Weather forecasting'], 'answer': 'Customer segmentation'}, {'question': 'What is the purpose of association in data analysis?', 'options': ['To group customers based on similarities', 'To find relationships between variables', 'To reduce the number of variables in data', 'To improve picture quality'], 'answer': 'To find relationships between variables'}, {'question': 'What is dimensional reduction used for?', 'options': ['Grouping customers based on age or location', 'Finding relationships between variables', 'Reducing the number of variables while preserving information', 'Analyzing market baskets'], 'answer': 'Reducing the number of variables while preserving information'}, {'question': 'What is a key difference between supervised and unsupervised learning models?', 'options': ['Supervised learning models require labeled data, while unsupervised learning models do not.', 'Unsupervised learning models are always more accurate than supervised learning models.', 'Supervised learning models work on their own to discover the inherent structure of data.', 'Unsupervised learning models predict outcomes based on labeled data.'], 'answer': 'Supervised learning models require labeled data, while unsupervised learning models do not.'}, {'question': 'Which type of learning model can automatically find patterns in data and group them together without human intervention?', 'options': ['Supervised learning models', 'Unsupervised learning models', 'Both supervised and unsupervised learning models', 'Neither supervised nor unsupervised learning models'], 'answer': 'Unsupervised learning models'}, {'question': 'What is a characteristic of unsupervised learning?', 'options': ['Handles large volumes of data in real time', 'Provides high transparency into data clustering', 'Requires a large amount of labeled data', 'Is ideal for medical images'], 'answer': 'Handles large volumes of data in real time'}, {'question': "What type of learning method is described as a 'happy medium' between supervised and unsupervised learning?", 'options': ['Reinforcement learning', 'Semi-supervised learning', 'Supervised learning', 'Unsupervised learning'], 'answer': 'Semi-supervised learning'}, {'question': 'What is a powerful way to gain data insights?', 'options': ['Manual analysis', 'Machine learning models', 'Random sampling', 'Survey methods'], 'answer': 'Machine learning models'}, {'question': 'What is the first step in choosing the right model for your data?', 'options': ['Deciding the type of data', 'Choosing between supervised and unsupervised learning', 'Collecting data', 'Evaluating data quality'], 'answer': 'Choosing between supervised and unsupervised learning'}]

# Define a function to get the chat completion
def get_chat_completion(messages):
    data = {
        "messages": messages,
        # "max_tokens": 500,
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }

    # Make the POST request to the API
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    # Handle the response
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

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
    # st.session_state.all_mcqs = dummy_mcqs
    st.session_state.all_mcqs = []

# Initialize OpenAI API
# st.session_state.bot = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
            {"role": "system", "content": f"Generate 2 MCQs from the following text in JSON format:\n{json.dumps(example_json, indent=4)}"},
            {"role": "user", "content": chunk}
        ]
        response = get_chat_completion(messages)
        print(response)
        try:
            mcqs = json.loads(response)
            if 'MCQs' in mcqs:
                all_mcqs.extend(mcqs['MCQs'])
                print("Extracted MCQs from 'MCQs' key.")
            elif 'questions' in mcqs:
                all_mcqs.extend(mcqs['questions'])
                print("Extracted MCQs from 'questions' key.")
            else:
                print(f"Unexpected format: {mcqs}")
        except KeyError as e:
            print(f"KeyError: {str(e)} - response did not have expected key.")
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {str(e)} - response could not be parsed.")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
        finally:
            # Continue regardless of errors
            continue
    
    return all_mcqs

test_text = """
welcome back to another machine learning explained video by assembly ai in this video we talk about supervised learning which is arguably the most important type of machine learning you will learn what it means examples of supervised learning or this data and training types of supervised learning and we touch on specific algorithms of supervised learning let's begin with the very basics what does machine learning mean machine learning is a sub-area of artificial intelligence and it's the study of algorithms that give computers the ability to learn and make decisions based on data and not from explicit instructions a popular example is learning to predict whether an email is spam or no spam by reading many different emails of these two types we typically differentiate between three types of machine learning supervised learning unsupervised learning and reinforcement learning in supervised learning the computer learns by making use of labeled data so we know the corresponding label or target of our data an example is again the spam prediction algorithm where we show many different emails to the computer and for each email we know if this was a spam email or not on the other hand in unsupervised learning the computer learns by making use of unlabeled data so we have data but we don't know the corresponding target an example is to cluster books into different categories on the basis of the title and other book information but not by knowing its actual category and then there is also reinforcement learning where so-called intelligent software agents take actions in an environment and automatically try to improve its behavior this usually works with a system of rewards and punishments and popular examples are games for example a computer can learn to be good in the snake game only by playing the game itself and every time 
it eats an apple or it dies it learns from this actions now in this video we are going to focus on supervised learning where we learn from labeled data now what is data data can be any relevant information we collect for our algorithm this can be for example user information like age and gender or text data or images or information within an image like measurements or color information the possibilities are endless here let's look at a concrete example in the popular iris flower data set we want to predict the type of iris flower based on different measurements we have 150 records of flowers with different attributes that have been measured before so for each flower we have the sepal 
length saypal width petal length and petal width these are called the features and we also have the corresponding species this is called the class the label or the target so this is a supervised case where we know the label we can 
represent this table in a mathematical way so we put each feature into a vector this is the feature vector and then we do this for all the different samples and when we do this for all the different samples we end up in a 2d representation which is also called a matrix additionally we can put all labels into one vector this is called the target vector now in supervised learning we take the features and the labels and show it to the computer so that it learns we call this the training step and the data we use is called the training data training is performed by specific algorithms that usually try to minimize an error during this training process and this is done by mathematical optimization methods which i won't go into more detail here after training we want to show new data to the computer that it has never seen before and where we don't know the label this is called our test data and now the trained computer should be able to make a decision based on the information it has seen and determine the correct target value and this is how supervised learning works there are two types of supervised learning classification and regression in classification we predict a discrete class label in the previous flower classification example our target values can only have the values 0 1 and 2 corresponding to the three different classes if we have more than two possible labels like here we call this a multi-class classification problem if we only have two labels usually zero and one is used then we call this a binary classification problem for example spam or no spam on the other hand in regression we try to predict a continuous target value meaning the target value can have a more or less arbitrary value one example is to predict house prices based on given information about the house and the neighborhood the target variable which is the price can basically have any value here now that we know what supervised learning means let's have a look at concrete algorithms i will not explain them in detail here i simply name them so that you have heard of them they all have a unique design and can be different in the way how it stores the information mathematically how it solves the training process through mathematical operations and how it transforms the data this list is not exhaustive but here are 10 algorithms that are nice to know some of them can be used for either regression or classification and some can even be used for both cases popular algorithms are linear regression logistic regression decision trees random forest naive bayes perceptron and multi-layer perceptron support vector machines or short svm k-nearest neighbors or short knn adaboost and neural networks which are part of the deep learning field alright i hope you enjoyed 
this video if you did so then please hit the like button and consider subscribing to the channel also if you want to try assembly ai for free then grab your free api token using the link in the description below and then i hope to 
see you in the next video bye
"""

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
            # time.sleep(5)
            st.info("Using test text instead.")
            return test_text
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
    if st.session_state.current_text == "" and st.session_state.all_mcqs == []:
        if uploaded_file is not None:
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_reader = PyPDFLoader("uploaded_file.pdf")
            documents = pdf_reader.load()
            pdf_text = "\n".join([doc.page_content for doc in documents])

            os.remove("uploaded_file.pdf")

            st.session_state.current_text = pdf_text
            st.session_state.all_mcqs = generate_mcqs_from_text(pdf_text)
            st.rerun()
        
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
        # Check if transcript and MCQs have not been generated yet
        if not st.session_state.get("current_text") and not st.session_state.get("all_mcqs", []):
            transcript_text = get_transcript(youtube_url)
            if transcript_text:
                st.session_state.current_text = transcript_text
                st.session_state.all_mcqs = generate_mcqs_from_text(transcript_text)
                st.rerun()

    # Check if there are any MCQs stored in session state
    all_mcqs = st.session_state.get("all_mcqs", [])
    if all_mcqs:
        current_question = st.session_state.get("current_question", 0)
        
        if current_question < len(all_mcqs):
            mcq = all_mcqs[current_question]

            st.write(f"**Question {current_question + 1}:** {mcq['question']}")
            
            # Generate a unique key using current question index and the question text to avoid duplication
            unique_key = f"question_{current_question}_{mcq['question']}"

            selected_option = st.radio("Select an option:", mcq['options'], key=unique_key)

            if st.button("Submit", key=f"submit_{current_question}"):
                st.session_state.selected_option = selected_option
                st.session_state.show_feedback = True

            if st.session_state.get("show_feedback", False):
                if st.session_state.selected_option == mcq['answer']:
                    st.success("Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"Wrong! The correct answer is: {mcq['answer']}")

                if st.button("Next", key=f"next_{current_question}"):
                    st.session_state.current_question += 1
                    st.session_state.show_feedback = False
                    st.session_state.selected_option = None
                    st.rerun()  # Refresh for the next question
        else:
            st.write(f"Quiz completed! Your score is {st.session_state.get('score', 0)} out of {len(all_mcqs)}.")
    else:
        st.write("Please enter a valid YouTube URL to generate a quiz.")
