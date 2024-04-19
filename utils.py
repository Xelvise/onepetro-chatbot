import os, time, json, streamlit as st
from collections import Counter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever 
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

memory = ConversationBufferWindowMemory(k=3, return_messages=True)

def retrieve_metadata(filepath:str):
    '''Extracts metadata from a text file\n
    Returns a dictionary containing the metadata'''
    data = dict()
    result = TextLoader(filepath).load()
    meta = result[0].page_content.split('\n----- METADATA END -----\n\n\n\n')[0].replace('----- METADATA START -----\n','')
    for section in meta.split('\n'):
        key = section.split(': ')[0]
        value = section.split(': ')[1]
        data[key] = value
    return data     # returns a dictionary containing the metadata


def create_doc(filepath:str):
    '''Creates a document object from a text file'''
    meta = retrieve_metadata(filepath)
    result = TextLoader(filepath).load()
    content = result[0].page_content.split('\n----- METADATA END -----\n\n\n\n')[1].strip()
    result[0].page_content = content
    result[0].metadata = meta
    return result[0]


def combine_docs(dir_path:str):
    '''Combines all text files in a given directory into a list of documents\n
    Returns a list of Document objects, a list of filenames and a list of metadata dictionaries'''
    path = list()
    filenames = list()
    # Iterate over all files in the dir_path and its subdirectories
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # Check if the file is a text file
            if file.endswith('.txt'):
                # Get the full path of the file
                file_path = os.path.join(root, file)
                path.append(file_path)    # Appends the filepaths to list
                filenames.append(file)   
    docs = [create_doc(file_path) for file_path in path]    # List of Document objects of each text file
    metadata = [retrieve_metadata(file_path) for file_path in path]     # List of Dictionary containing metadata for each file
    return docs, filenames, metadata


def split_to_chunks(list_of_docs):
    '''Splits each doc into chunks of 1000 tokens with an overlap of 200 tokens'''
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['.\n\n\n\n','.\n\n'],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(list_of_docs)
    return chunks


def vectorize_doc_chunks(doc_chunks, index_name:str='spenaic-papers', partition:str=None):
    '''Embeds each chunk of text and store in Pinecone vector db'''
    length = len(doc_chunks)
    split = round(length/4)
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=partition, pinecone_api_key=os.getenv('PINECONE_APIKEY_CONTENT'))     # initialize connection to Pinecone vectorstore
    _ = vector_store.add_documents(doc_chunks[:split])
    time.sleep(120)
    _ = vector_store.add_documents(doc_chunks[split:2*split])
    time.sleep(120)
    _ = vector_store.add_documents(doc_chunks[2*split:3*split])
    time.sleep(120)
    _ = vector_store.add_documents(doc_chunks[3*split:])


def vectorize_paper_titles(dir_path:str, index_name:str='paper-title'):
    '''Embeds each paper title and store in a Pinecone vector db'''
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    _, filenames, metadata = combine_docs(dir_path)
    docs = list()
    for name, meta in zip(filenames, metadata):
        mydoc = Document(page_content=name.replace('.txt',''), metadata={'Authors':meta['Authors'], 'Publication year':int(meta['Publication Date'].split(' ')[1]), 'ref link':meta['Reference Link']})
        docs.append(mydoc)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_APIKEY_TITLE'))
    vectorstore.add_documents(docs)


@st.cache_data(show_spinner=False)
def find_similar_papers(paper_title:str, k:int=10, year:int=None, index_name:str='paper-title') -> list:
    '''`date`: `'month YYYY'`\n
        Uses similarity search to retrieve 10 most-similar papers according to a given paper title\n
        Where year is given, metadata filtering is applied to further narrow down the search from selected year till present
    '''
    papers = list(); ref_links = list()
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_APIKEY_TITLE'))
    if year != None:
        # Limit the search to only include k-papers from the given year till present
        docs = vectorstore.similarity_search(paper_title, k=k, filter={'date': {"$gte": f"{year}"}})
        if docs == []:    # If no papers are found, return an empty list
            return docs     
        for doc in docs:
            papers.append(f"{doc.page_content} (Year: {str(int(doc.metadata['Publication year']))})")
            ref_links.append(doc.metadata['ref link'])
    else:
        # Retrieve k-papers from the entire database
        docs = vectorstore.similarity_search(paper_title, k=k)
        if docs == []:     # If no papers are found, return an empty list
            return docs
        for doc in docs:
            papers.append(f"{doc.page_content} (Year: {str(int(doc.metadata['Publication year']))})")
            ref_links.append(doc.metadata['ref link'])
    return papers, ref_links


@st.cache_data(show_spinner=False)
def summarize_paper(paper_title:str, year:str):
    '''Summarizes a paper, when given its title'''

    doc = [create_doc(os.path.join(os.getcwd(), 'files', year, paper_title+'.txt'))]
    metadata = retrieve_metadata(os.path.join(os.getcwd(), 'files', year, paper_title+'.txt'))
    prompt_template = """Write an elaborate summary of the given text. Ensure to highlight key points that could be insightful to the reader.\n
        Text: "{text}"
        ELABORATE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.5)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    doc_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return doc_chain.run(doc), metadata


@st.cache_data(show_spinner=False)
def get_response(query:str):
    '''Generates a response to User query, while also providing a list of similar papers'''
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    # llm = GoogleGenerativeAI(google_api_key=os.getenv('GOOGLE_API_KEY'), model='gemini-pro', temperature=0.7)
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.7)

    # initialize the vector store object
    vectorstore = PineconeVectorStore(
        index_name='spenaic-papers',
        embedding=embeddings,
        pinecone_api_key=os.getenv('PINECONE_APIKEY_CONTENT')
    ).as_retriever(      # Only retrieve documents that have a relevance score of 0.8 or higher
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold':0.8, 'k':5}
    )
    
    doc_retrieval_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # create a runnable that, when invoked, retrieves List[Docs] based on user_input and chat_history
    doc_retriever_runnable = create_history_aware_retriever(llm, vectorstore, doc_retrieval_prompt)

    elicit_response_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the human's questions based on the given context ONLY. But if you cannot find an answer based on the context, you should either request for additional context or, if it is a question, simply say - 'I have no idea.':\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    # create a runnable that, when invoked, appends retrieved List[Docs] to prompt and passes it on to the LLM as context for generating response to user_input
    context_to_response_runnable = create_stuff_documents_chain(llm, elicit_response_prompt)

    # chains up two runnables to yield the final output that would include user_input, chat_history, context and answer
    retrieval_chain_runnable =  create_retrieval_chain(doc_retriever_runnable, context_to_response_runnable)     # chains up two runnables to yield the final output that would include human_query, chat_history, context and answer

    response = retrieval_chain_runnable.invoke({
        "chat_history": memory.load_memory_variables({})['history'],
        "input": query
    })
    # since the memory is a buffer window, we append to the buffer the query and answer of the current conversation
    memory.save_context({"input": f"{response['input']}"}, {"output": f"{response['answer']}"})

    for phrase in ['I don\'t know','AI assistant','I apologize','Feel free to share','more context','You\'re welcome!','I do not know','I have no idea','provided context',"I couldn't ",'I cannot',"If you have any more questions","I appreciate","How can I help you today"]:
        if phrase in response['answer']:
            return response['answer'], ''
        
    papers = [docs.metadata['Title'] for docs in response['context']]      # Extracts the title of each paper from the context
    most_frequent_paper = Counter(papers).most_common(1)[0][0]      # Extracts the most frequent paper title from the context
    paper_titles, links = find_similar_papers(most_frequent_paper, k=7)     # Finds similar papers based on the most frequent paper title

    return response['answer'], list(zip(paper_titles, links))


def save_conversation_history():
    chat_history = {
        'messages': st.session_state.get('messages', [{"role":"assistant", "content":"Hello, there.\n How may I assist you?"}])
    }
    with open('chat_history.json', 'w') as f:
        json.dump(chat_history, f)


def load_conversation_history():
    try:
        with open('chat_history.json', 'r') as f:
            chat_history = json.load(f)
            st.session_state.messages = chat_history.get('messages', [{"role":"assistant", "content":"Hello, there.\n How may I assist you?"}])
    except FileNotFoundError:
        st.session_state.messages = [{"role":"assistant", "content":"Hello, there.\n How may I assist you?"}]



# Replace YEAR with the specific date folder containing the text files to be vectorized
YEAR = '2023'

if __name__ == '__main__':
    # YOU SHOULD RUN THIS SCRIPT ONLY WHEN YOU HAVE NEWER TEXT FILES THAT HASN'T BEEN EMBEDDED OR/AND STORED IN PINECONE
    list_of_docs,_,_ = combine_docs(os.path.join(os.getcwd(), 'files', YEAR))
    doc_chunks = split_to_chunks(list_of_docs)
    vectorize_doc_chunks(doc_chunks)
    vectorize_paper_titles(os.path.join(os.getcwd(), 'files', YEAR))
