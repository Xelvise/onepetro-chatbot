import streamlit as st
from streamlit_float import float_init, float_parent
from utils import *
st.set_page_config(page_title='Chat with papers', page_icon='📚')

st.markdown('### Here, Research papers are made Conversational and Insightful 📚')

with st.sidebar:
    float_init()
    with st.container():
        if st.button('Clear Chats', use_container_width=True):
            if os.path.exists('chat_history.json'):
                os.remove('chat_history.json')
                del st.session_state.messages
                st.rerun()
            else:
                pass
        float_parent("bottom: 2%;")

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
    st.toast('thinking...'); time.sleep(4); st.toast('give me few secs...')#; time.sleep(2); st.toast('a couple more, please...')
    response, similar_papers = get_response(query=prompt)
    sources = ''
    if similar_papers != '':
        for (title, link) in similar_papers:
            sources += f'<a href="{link}" style="text-decoration: none;">{title}</a>\n\n'
        st.chat_message('assistant').write(response + '\n\n**Check out related papers:**\n\n' + sources, unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant", "content":response + '\n\n**Check out related papers:**\n\n' + sources})
    else:
        st.chat_message('assistant').write(response)
        st.session_state.messages.append({"role":"assistant", "content":response})

save_conversation_history()
