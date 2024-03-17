import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate


openai.api_key = st.secrets["OPENAI_API_KEY"]


@st.cache_resource
def load_chain():
    """
    The 'load_chain()' function initializes and configures a conversational
    retrieval chain for answering user questions.
    :return: A ConversationalRetrievalChain object.
    """

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()
    
    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0)

    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Create memory 'chat_history'
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")
    
    # Create system prompt
    template = (
        "You are an AI assistant for answering questions about the Company "
        "in-a-Box Notion Template.\n"
        "You are given the following extracted parts of a long document and a question. "
        "Provide a conversational answer.\n"
        "If you don't know the answer, just say 'Sorry, I don't know ... 😔'. "
        "Don't try to make up an answer.\n"
        "If the question is not about the Company in-a-Box Notion Template, politely inform "
        "the user that you are tuned to only answer questions about the Company in-a-Box Notion Template.\n"
        "\n"
        "{context}\n"
        "Question: {question}\n"
        "Helpful Answer:"
    )

    # Initialize the ConversationalRetrievalChain with your components
    chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0),
                                                  retriever=retriever,
                                                  memory=memory,
                                                  get_chat_history=lambda h: h,
                                                  verbose=True)
    
    # Configure the prompt for the conversation chain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
    
    return chain

# def load_index_bytes():
#     # This function should contain logic to load your serialized FAISS index bytes from storage.
#     # For example, you might read from a binary file:
#     # return open('path_to_your_serialized_faiss_index', 'rb').read()
#     pass