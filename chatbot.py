from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import docx2txt
import os
import openai

openai_api_key = "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"

# Custom CSS styles
CUSTOM_CSS = """
<style>
body {
    background-color: #000000; /* Black background */
    color: #ffffff; /* White text color */
    font-family: 'Arial', sans-serif; /* Elegant font */
}

.stButton button {
    background-color: #ffd700; /* Gold button background */
    color: #000000; /* Black text color for buttons */
}

.stTextInput input, .stTextArea textarea {
    background-color: ##2b3332; /* White input background */
    color: 	#FFFFFF; /* Black text color for inputs */
}

.stMarkdown {
    color: #ffd700; /* Gold text color for markdown elements */
}
</style>
"""


def read_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def read_docx(docx_file):
    text = docx2txt.process(docx_file)
    return text


chat_history = []


def main():
    load_dotenv()
    st.set_page_config(page_title="Lawyer Bot")

    # Load the custom CSS styles
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Add the logo and OracleDelphAI branding at the top center
    st.image("logo.png", width=350)  # Adjust the width to make the logo smaller
    st.markdown('<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">'
                '<h1 style="text-align: center;">OracleDelphAI - Lawyer Bot 💬</h1>'
                '</div>', unsafe_allow_html=True)

    # upload file
    uploaded_file = st.file_uploader("Upload your file (PDF, DOCX or DOC)", type=["pdf", "docx", "doc"])

    # extract the text
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".pdf":
            text = read_pdf(uploaded_file)
        elif file_extension == ".docx":
            text = read_docx(uploaded_file)
        elif file_extension == ".doc":
            # Note: Handling .doc files might require additional libraries like "pywin32"
            st.error("Sorry, handling .doc files requires additional libraries and might not work in this web app.")
            return
        else:
            st.error("Unsupported file format. Please upload a PDF, DOCX or DOC file.")
            return

        # split into chunks
        char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                                   chunk_overlap=200, length_function=len)
        text_chunks = char_text_splitter.split_text(text)

        # Pad the text chunks to have the same length
        max_chunk_length = max(len(chunk) for chunk in text_chunks)
        text_chunks_padded = [chunk + " " * (max_chunk_length - len(chunk)) for chunk in text_chunks]

        # create embeddings
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(text_chunks_padded, embeddings)  # Use the padded chunks

        llm = OpenAI(model_name="text-davinci-003")
        chain = load_qa_chain(llm, chain_type="stuff")

        # Show user input and chat history in the chatbox
        st.sidebar.header("Chat")
        chatbox = st.sidebar.text_area("Type your question:")

        if st.sidebar.button("Ask"):
            docs = docsearch.similarity_search(chatbox)
            context = "\n".join([f"{sender}: {message}" for sender, message in chat_history])
            response = chain.run(input_documents=docs, question=chatbox + "\n" + context)

            # Add user input and chatbot response to the chat history
            chat_history.append(("You", chatbox))
            chat_history.append(("LawBOT", response))

        # Display the chat history in the sidebar
    st.sidebar.header("Chat History")
    for sender, message in chat_history:
        st.sidebar.write(sender + ":")
        st.sidebar.write(message)

    st.sidebar.markdown(
        "---\n\n*LawBOT is an AI-powered chatbot designed to help with legal documents.*"
    )


if __name__ == '__main__':
    main()
