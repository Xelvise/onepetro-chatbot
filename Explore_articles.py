import streamlit as st
from utils import *
import datetime, os, json, time
import calendar

if 'clicked_search' not in st.session_state:
    st.session_state.clicked_search = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = False
if 'summary' not in st.session_state:
    st.session_state.summary = False

def change_state():
    st.session_state.clicked_search = True


st.set_page_config(page_title='Explore articles', page_icon='📚')
st.subheader('Explore varieties of papers present in SPENAIC\'s Catalog 📚')
st.divider()

# search container
with st.container():    
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        query = st.text_input('Search for articles', key='search_query', placeholder='Search for articles...')
    with col2:
        date = st.selectbox('Presentation date', options=['August 2024'], index=None, placeholder='Select a date', key='query_date')
    st.button('Search', on_click=change_state)
st.divider()

# define logic for search
if st.session_state.clicked_search:
    if query:
        st.toast('Fetching similar articles...')
        search_result = find_similar_papers(paper_title=query)      # returns a list of lists containing paper title and presentation date
        if not search_result:
            st.toast('No search result was found')
        st.session_state.search_results = search_result
        st.session_state.clicked_search = False

    elif query and date:
        st.toast('Fetching similar articles...')
        search_result = find_similar_papers(paper_title=query, date=date)
        if not search_result:
            st.toast('No search result was found')
        st.session_state.search_results = search_result
        st.session_state.clicked_search = False

    else:
        st.toast('Enter a search query to get started')
        st.session_state.clicked_search = False

with st.sidebar:
    if st.session_state.search_results:
        options = st.selectbox('View similar papers', options=st.session_state.search_results, index=None, placeholder='Select a paper')
        if st.button('Choose paper'):
            if options:
                st.session_state.selected_paper = True
            else:
                st.toast('No paper was selected')
    else:
        st.selectbox('View similar papers', disabled=True, index=None, placeholder='No search results found', options=[None])

if st.session_state.selected_paper:
    paper_title = options.split('(Date:')[0].strip()
    paper_date = options.split('(Date:')[1].replace(')', '').strip()
    st.session_state.summary, metadata = summarize_paper(paper_title, paper_date)
    for key, value in metadata.items():
        st.write(f"#### {key}: {value}")
    st.text('')
    st.write('##### A Brief Overview: ')
    st.write(st.session_state.summary)
    st.session_state.selected_paper = False

    