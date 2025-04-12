import os
import utils
import streamlit as st
from streaming import StreamHandler
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with your Documents')
st.write('Has access to custom documents and can respond to user queries by referring to the content within those documents')

class CustomDataChatbot:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-4o-mini"

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_files):
        # Initialize progress bar for document processing in the sidebar
        total_files = len(uploaded_files)
        progress_bar = st.sidebar.progress(0, text=f"Processing {total_files} files...")

        # Load documents
        docs = []
        for idx, file in enumerate(uploaded_files):
            file_path = self.save_file(file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

            # Update progress bar for file upload in the sidebar
            progress_value = (idx + 1) / total_files
            progress_text = f"Processed {idx + 1} out of {total_files} files..."
            progress_bar.progress(progress_value, text=progress_text)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever()

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model,
                         temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True)
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []

        # User Inputs
        uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)

        if uploaded_files:
            current_file_names = [file.name for file in uploaded_files]
            previous_file_names = [file.name for file in st.session_state.uploaded_files]

            if set(current_file_names) != set(previous_file_names):
                st.session_state.uploaded_files = uploaded_files
                qa_chain = self.setup_qa_chain(uploaded_files)
                st.session_state.qa_chain = qa_chain
        else:
            st.error("Please upload PDF documents to continue!")
            st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query and 'qa_chain' in st.session_state:
            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                try:
                    st_cb = StreamHandler(st.empty())
                    response = st.session_state.qa_chain.run(user_query, callbacks=[st_cb])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
                except Exception as e:
                    print(e)

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()