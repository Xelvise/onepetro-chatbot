import streamlit as st
from utils import *
from logger import logging
from datetime import datetime

present_year = datetime.now().year
years = list(range(1998, present_year))
timeframe = [str(year)+' till present' for year in years]

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

st.set_page_config(page_title='Explore papers', page_icon='📚')
st.subheader('Explore varieties of papers present in OnePetro\'s Catalog')
st.divider()

# search container
with st.container():    
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        query = st.text_input('Search for papers', key='search_query', placeholder='Search for papers...')
    with col2:
        year = st.selectbox('Publication year', options=timeframe, index=None, placeholder='Select timeframe', key='query_date')
    st.button('Search', on_click=change_state)
st.divider()

# define logic for search
if st.session_state.clicked_search:
    if query:
        st.toast('Fetching similar papers...')
        search_result, _ = find_similar_papers(paper_title=query)      # returns a list of lists containing paper title and presentation date
        if not search_result:
            st.toast('No search result was found')
        st.session_state.search_results = search_result
        st.session_state.clicked_search = False

    elif query and year:
        year = int(year.replace(' till present',''))
        st.toast('Fetching similar papers...')
        search_result, _ = find_similar_papers(paper_title=query, year=year)
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
                st.session_state.selected_paper = options
            else:
                st.toast('No paper was selected')
    else:
        st.selectbox('View similar papers', disabled=True, index=None, placeholder='No search results found', options=[None])

if st.session_state.selected_paper:
    paper_title = st.session_state.selected_paper.split('(Year:')[0].strip()
    paper_year = st.session_state.selected_paper.split('(Year:')[1].replace(')', '').strip()
    # try:
    st.session_state.summary, metadata = summarize_paper(paper_title, paper_year)
    # except RuntimeError as e:
    #     st.toast('An error occurred while fetching the paper. Please try another')
    #     logging.error(e)
    #     st.stop()
    for key, value in metadata.items():
        st.write(f"**{key}** - {value}")
    st.text('')
    st.write(st.session_state.summary)
