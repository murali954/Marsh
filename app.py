import streamlit as st
from PyPDF2 import PdfReader
import cohere

# Set your Cohere API key directly
cohere_api_key = "26Xw62woNXPk1SfiQh79vNXhcIzHoiUHF0B5vSJm"
co = cohere.Client(cohere_api_key)

# Function to get text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to handle user input and interaction with Cohere
def user_input(user_question, context):
    try:
        response = co.generate(
            model='command',  # Use a valid model ID from Cohere
            prompt=f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:",
            max_tokens=150
        )
        answer = response.generations[0].text.strip()
        st.write("User Question:", user_question)
        st.write("Answer:", answer)
    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Cohere")

    context = ""

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    context = raw_text  # Update context with actual PDF content
                    st.text_area("Extracted PDF Content", context, height=300)
                    st.success("Processing complete")
                else:
                    st.warning("No text extracted from PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and context:
        user_input(user_question, context)
    elif user_question:
        st.warning("Please upload and process a PDF first")

if __name__ == "__main__":
    main()
