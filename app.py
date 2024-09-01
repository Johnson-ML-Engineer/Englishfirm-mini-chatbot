import os
import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Document Processing
def read_documents(directory):
    return PyPDFDirectoryLoader(directory).load()

def chunk_data(docs, chunk_size=800, chunk_overlap=40):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)

# RAG System Setup
def create_embeddings_and_store(doc_chunks):
    vectorstore = Chroma.from_documents(
        documents=doc_chunks, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    system_prompt = """
You are an AI assistant for question-answering tasks about Englishfirm. Englishfirm is one of the leading PTE coaching academies in Sydney, distinguished for providing 100% one-on-one coaching, a unique offering among the 52 PTE institutes in Sydney. 
Englishfirm operates 7 days a week from two branches: Sydney CBD (Pitt Street) and Parramatta. 
The key team members include Nimisha James (Head Trainer), Avanti (Associate Trainer), Vandana (Trainer), and Kaspin (Student Counsellor for University Admissions).alyze the provided context and answer the user's question concisely. Follow these guidelines:

1. Utilize only the information provided in the context above to formulate your responses.
2. If the context doesn't contain sufficient information to answer a question, respond with: "I don't have enough information to answer this question."
3. Craft clear, direct answers limited to a maximum of seven sentences.
4. Maintain a professional and informative tone in all interactions.
5. Highlight Englishfirm's unique features when relevant, such as the exclusive one-on-one coaching and convenient locations.
Context:
{context}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=250)
    
    question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Chatbot Response
def chatbot_response(query, rag_chain):
    try:
        return rag_chain.invoke({"input": query})["answer"]
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Initialize RAG Chain
def initialize_rag_chain():
    try:
        documents = read_documents("path/to/your/pdf/directory")
        doc_chunks = chunk_data(documents)
        return create_embeddings_and_store(doc_chunks)
    except Exception as e:
        st.error(f"Error initializing RAG chain: {str(e)}")
        return None

# Streamlit App
def main():
    st.set_page_config(page_title="Mini AI Bot for Englishfirm.com", page_icon="🤖", layout="wide")
    st.title("Mini AI Bot for Englishfirm.com")
    st.sidebar.info("Englishfirm is the one of the best PTE coaching academies in Sydney.  Among 52 PTE institutes in Sydney, Englishfirm is the only training centre in Sydney that offers 100% one-on-one coaching. Englishfirm has 2 branches in Sydney, operating 7 day a week. We operate from Sydney CBD campus (Pitt Street) and Parramatta.")
    st.sidebar.warning("Disclaimer: This chatbot provides information related to Englishfirm's IELTS, PTE, and Spoken English coaching services. While we strive to offer accurate and up-to-date information, this should not be considered a substitute for professional educational advice. For personalized guidance and assistance, please contact our experts directly.")
    
    rag_chain = initialize_rag_chain()
    
    if rag_chain is None:
        st.error("Failed to initialize the chatbot. Please try again later.")
        return
    
    st.write("Welcome to Englishfirm! I'm here to assist you with any questions regarding our IELTS, PTE, and Spoken English coaching. How can I help you today?")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                response = chatbot_response(prompt, rag_chain)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    api_key = "AIzaSyAyS_Bb-vri5grDlsEdvqqtDHXRVvc3gDw"
    os.environ["GOOGLE_API_KEY"] = api_key
    main()
