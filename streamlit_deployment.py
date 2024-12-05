import os
import streamlit as st
import openai
from pydub import AudioSegment
# import bookrecommendation
import pandas as pd

from openai import OpenAI

from pinecone import Pinecone
# Assuming you have the secrets manager setup, otherwise replace with your own method for securing API keys
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"
client = OpenAI()

# nhúng api key của pinecone và index vào
pc = Pinecone(api_key="YOUR PINECONE API KEY")
index_books = pc.Index("booksv2")
index_movies = pc.Index("movies")

# Set the page config to wide mode
st.set_page_config(layout="wide")

# Title of the application
st.title('OpenAI API Applications')

# Sidebar for navigation
st.sidebar.title("Applications")
applications = ["Blog Generation", "Book Recommendation", "Generate Image", "Mental Health FAQ", "Voice to text"]
application_choice = st.sidebar.radio("Choose an Application", applications)

def blog_generation(topic, additional_pointers):
    prompt = f"""
    You are a copy writer with years of experience writing impactful blog that converge and help elevate brands.
    Your task is to write an outline of a  blog on any topic system provides you with. Make sure to write in a format that works for Medium. 
    Outline should have 3 parts: introduction, details, conclusion.
    Your respone must be below 150 words.

    Topic: {topic}
    Additiona pointers: {additional_pointers}
    """

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=1,
        max_tokens=150,
    )

    return response.choices[0].text.strip()

def generate_image(prompt, number_of_images=1):
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="256x256",
        n=number_of_images,
    )

    return response

def recommendation(client_input_text,index):
    user_vector = client.embeddings.create(
        model="text-embedding-3-small",
        input=client_input_text
        )

    # top_k = 10
    # send to pinecone
    result_vector = user_vector.data[0].embedding

    result = index.query(
        vector=result_vector,
        top_k=10,
        include_metadata = True
        )

    return result

def faq(dataset_str):   
    # OpenAI setup
    messages_buffer = [{"role": "user", "content": """I want you to act as a support agent. Your name is "My Super Assistant". 
                    You will provide me with answers from the given info. If the answer is not included, say exactly "Ooops! I don't know that." and stop after that. Refuse to answer any question not about the info. Never break character."""}]
    messages_buffer.append({"role": "user", "content":dataset_str})

    user_input = st.text_input("Your Question:", placeholder="Type your question here...")
    
    if st.button("Get Answer"):
        if user_input.lower() == "exit":
            st.write("Goodbye!")
            return
    
    messages_buffer.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_buffer,
            temperature=0,
            max_tokens=100,
        )
        assistant_response = response.choices[0].message.content
        messages_buffer.append({"role": "assistant", "content": assistant_response})
        st.write(f"**Assistant:** {assistant_response}")
    except Exception as e:
        st.error(f"Error communicating with OpenAI API: {e}")

# healthfaq = getDataset("Mental_Health_FAQ.csv")

# audio_file = open("audio-file-whisper.mp3", "rb")

def uploadMp3():
    """
    Function to upload an audio file and return it as an MP3 file object.
    """
    st.header("Upload Audio File")
    st.write("Upload an audio file. It will be converted and returned as an MP3 file.")

    # Allow the user to upload audio files
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "ogg", "flac", "mp3"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        # Check if conversion is needed
        if not uploaded_file.name.endswith(".mp3"):
            # Convert to MP3 using pydub
            audio = AudioSegment.from_file(temp_file_path)
            mp3_file_path = "converted_audio.mp3"
            audio.export(mp3_file_path, format="mp3")
            os.remove(temp_file_path)  # Clean up the temporary file
            return open(mp3_file_path, "rb")  # Return the MP3 file object
        else:
            # File is already MP3, return it directly
            return open(temp_file_path, "rb")
    
    else:
        st.warning("No file uploaded.")
        return None

def uploadExcel():
    """
    Function to upload a dataset (CSV or Excel) and process it to extract and format 
    the 'Questions' and 'Answers' columns.
    """
    st.header("Upload Dataset File")
    st.write("Upload a dataset file (CSV or Excel) containing 'Questions' and 'Answers' columns.")

    # Allow the user to upload a CSV or Excel file
    uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Read the file into a Pandas DataFrame
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            
            # Ensure the necessary columns exist
            if not {'Questions', 'Answers'}.issubset(df.columns):
                st.error("The dataset must contain 'Questions' and 'Answers' columns.")
                return None
            
            # Extract the required columns and read the first 20 rows
            subset = df[['Questions', 'Answers']]
            small_dataset = subset.iloc[:20]

            # Format the dataset into a string
            dataset_str = "Document content:\n"
            for _, row in small_dataset.iterrows():
                dataset_str += f"Question: {row['Questions']} Answer: {row['Answers']}\n"

            st.success("File uploaded and processed successfully!")
            # st.text(dataset_str)  # Display the formatted dataset string
            
            return dataset_str  # Return the formatted string for further use
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            return None
    else:
        st.warning("No file uploaded.")
        return None

def voiceToText(audio_file):
    transcript_translated = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    return transcript_translated.text

# Main application logic
def main():
    if application_choice == "Blog Generation":
        st.header("Text Completion with GPT-3")
        st.write("Input some text and get a completion.")
        input_text = st.text_area("Enter text here:")
        additional_pointers = st.text_area("Enter additional pointers here:")
        
        if st.button("Complete Text"):
            with st.spinner('Generating...'):
                completion = blog_generation(input_text, additional_pointers)
                st.text_area("Generated blog:", value=completion, height=200)

    elif application_choice == "Generate Image":
        st.header("Image Generation with DALL-E")
        st.write("Input some text and generate an image.")
        input_text = st.text_area("Enter text for image generation:")

        number_of_images = st.slider("Choose the number of images to generate", 1, 5, 1) 
        if st.button("Generate Image"):
            
            outputs = generate_image(input_text, number_of_images)
            for output in outputs.data:
                st.image(output.url)

    elif application_choice == "Book Recommendation":
        st.header("Book Recommendation with GPT")
        st.write("Input a book description and get a recommendation.")

        input_text = st.text_area("Enter book description:")

        if st.button("Get Book") and input_text != "":
            with st.spinner('Generating...'):
                result = recommendation(input_text,index_books)

                # Access the 'matches' key from the response
                if 'matches' in result:
                    st.write("Here are top 10 books based on your input:")
                    for match in result['matches']:
                        st.write(match['metadata']['title'])
                else:
                    st.write("No matches found!")
        
    elif application_choice == "FAQ":
        st.header("FAQ")
        st.write("Input a question and get an answer.")
        healthfaq = uploadExcel()
        if healthfaq is not None:
            faq(healthfaq)
        
    elif application_choice == "Voice to text":
        st.header("Voice to text")
        st.write("Upload Audio File: ")
        # audio_file = file_upload()
        # transcript = voiceToText(audio_file)
        # st.write("Voice to translate: ")
        # st.write(transcript)
        audio = uploadMp3()
        if st.button("Complete Text"):
            transcript = voiceToText(audio)
            st.write("Voice to translate: ")
            st.write(transcript)
            
# Run the main function
if __name__ == "__main__":
    main()

