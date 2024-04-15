import streamlit as st
from utils import *
st.set_page_config(page_title='Chat with articles', page_icon='📚')

st.markdown('### Here, Research papers are made Conversational and Insightful 📚')


# initialize chat history
if 'messages' not in st.session_state:
    load_conversation_history()

# display chat messages from history on app rerun
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar='img/420bb535f6bd4081a8cb4308e95f9768.jpg'):
            st.write(msg["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(msg["role"]):
            st.write(msg["content"], unsafe_allow_html=True)
            
# accept user input
if prompt := st.chat_input(placeholder='Enter your prompt...'):
    st.session_state.messages.append({"role":"user", "content":prompt})     # Add user msg to chat history
    st.chat_message('user', avatar='img/420bb535f6bd4081a8cb4308e95f9768.jpg').write(prompt)   # Display user msg in chat msg container
    st.toast('thinking...'); st.toast('give me few secs...')#; time.sleep; st.toast('a couple more, please...')
    response, metadata = get_response(query=prompt)
    if metadata != '':
        st.chat_message('assistant').write(response + '\n\n**SOURCE:**\n\n' + metadata)
        st.session_state.messages.append({"role":"assistant", "content":response + '\n\nSOURCE:\n\n' + metadata})
    else:
        st.chat_message('assistant').write(response)
        st.session_state.messages.append({"role":"assistant", "content":response})

save_conversation_history()
