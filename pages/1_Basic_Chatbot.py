import utils
import streamlit as st
from streaming import StreamHandler

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Basic Chatbot')
st.write('Allows users to interact with the OpenAI LLMs')


class Basic:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-4o-mini"

    def setup_chain(self):
        llm = ChatOpenAI(model_name=self.openai_model,
                     temperature=0, streaming=True)
        chain = ConversationChain(llm=llm, verbose=True)
        return chain

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                try:
                    st_cb = StreamHandler(st.empty())
                    response = chain.run(user_query, callbacks=[st_cb])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
                except Exception as e:
                    print(e)



if __name__ == "__main__":
    obj = Basic()
    obj.main()
