import os, time, json, streamlit as st
from collections import defaultdict
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

def retrieve_metadata(filepath:str):
    '''Extracts metadata from a text file'''
    data = dict()
    result = TextLoader(filepath).load()
    content = result[0].page_content.split('\n----- METADATA END -----\n\n\n\n')[1].strip()
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


def combine_docs(directory):
    '''Combines all text files in a subdirectory into a list of documents'''
    path = list()
    # Iterate over all files in the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a text file
            if file.endswith('.txt'):
                # Get the full path of the file
                file_path = os.path.join(root, file)
                path.append(file_path)
    # Append the file paths to list
    docs = [create_doc(file_path) for file_path in path]
    return docs


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
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))    # initializes the OpenAI embeddings
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=partition, pinecone_api_key=os.getenv('PINECONE_APIKEY_CONTENT'))     # initialize connection to Pinecone vectorstore
    _ = vector_store.add_documents(doc_chunks[:split])
    time.sleep(120)
    _ = vector_store.add_documents(doc_chunks[split:2*split])
    time.sleep(120)
    _ = vector_store.add_documents(doc_chunks[2*split:3*split])
    time.sleep(120)
    _ = vector_store.add_documents(doc_chunks[3*split:])


def vectorize_paper_titles(path:str, index_name:str='paper-title'):
    '''Embeds each paper title and store in Pinecone vector db'''
    list_of_docs = combine_docs(path)
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    docs = list()
    for doc in list_of_docs:
        for key, value in doc.metadata.items():
            if key == 'Title':
                mydoc = Document(page_content=value, metadata={'Authors':doc.metadata['Authors'], 'Presentation date':doc.metadata['Publication Date']})
                docs.append(mydoc)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_APIKEY_TITLE'))
    vectorstore.add_documents(docs)


@st.cache_data(show_spinner=False)
def find_similar_papers(paper_title:str, date:str=None, index_name:str='paper-title') -> list:
    '''`date`: `'month YYYY'`\n
        Uses similarity search to retrieve 5 most-similar papers according to a given paper title\n
        Where date is given, metadata filtering is applied to further narrow down the search
    '''
    papers = list()
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_APIKEY_TITLE'))
    if date != None:
        docs = vectorstore.similarity_search(paper_title, k=5, filter={'date': {"$eq": f"{date}"}})
        if docs == []:
            return docs
        for doc in docs:
            papers.append(f"{doc.page_content}  (Date: {doc.metadata['Presentation date']})")
    else:
        docs = vectorstore.similarity_search(paper_title, k=5)
        if docs == []:
            return docs
        for doc in docs:
            papers.append(f"{doc.page_content}  (Date: {doc.metadata['Presentation date']})")
    return papers


@st.cache_data(show_spinner=False)
def summarize_paper(paper_title:str, date:str):
    '''Summarizes a paper, when given its title'''

    doc = [create_doc(os.path.join(os.getcwd(), 'files', date, paper_title+'.txt'))]
    metadata = retrieve_metadata(os.path.join(os.getcwd(), 'files', date, paper_title+'.txt'))
    prompt_template = """Write an elaborate summary of the given Text. Ensure to highlight key points that could be insightful to the reader.\n
        Text: "{text}"
        ELABORATE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.7)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    doc_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return doc_chain.run(doc), metadata


@st.cache_data(show_spinner=False)
def get_response(query:str, paper_title:str=None):
    '''Generates a response to a query, with/without a paper title'''
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    llm = ChatGoogleGenerativeAI(google_api_key=os.environ['GOOGLE_API_KEY'], model='gemini-pro', temperature=0.7, convert_system_message_to_human=True)

    # This controls how the standalone question is generated.
    prompt = PromptTemplate.from_template(
        '''Combine the chat history and user question into a search query to look up in order to get information relevant to the conversation.
            If the answer to User question is not in the provided context, simply say - 'My apologies! I'm just an Energy Industry Chatbot that can respond to queries related to the Energy Industry'\n
            Chat History: {chat_history}\n
            User Question: {question}'''
    )

    if paper_title != None:
        # initialize the vector store object
        vectorstore = PineconeVectorStore(
            index_name='spenaic-papers',
            embedding=embeddings,
            pinecone_api_key=os.getenv('PINECONE_APIKEY_CONTENT')
        ).as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold':0.8, 'k':5, 'filter':{'Title':paper_title}}
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=ConversationBufferWindowMemory(k=3, memory_key='chat_history', output_key='answer', return_messages=True),
            retriever=vectorstore,
            condense_question_prompt=prompt,
            return_source_documents=True,
            response_if_no_docs_found="I'm sorry, I couldn't find any relevant documents."
        )
        response = chain.invoke({'question': query})
        answer = response['answer']
        for phrase in ['I don\'t know','I do not know','provided context','I cannot find','I can\'t find','I\'m just an Energy Industry Chatbot']:
            if phrase in answer:
                return answer, ''

        # Merging all retrieved metadata into one, while retaining only the unique values
        merged_metadata = defaultdict(list)     # creates a dictionary with default value as a list
        for doc in response['source_documents']:    # iterate over the source documents
            for k, v in doc.metadata.items():   # iterate over the key-value pair in the metadata of current document
                merged_metadata[k].append(v)
        meta = {k: list(set(v))[0] for k, v in merged_metadata.items()}    # removes duplicates from the metadata
        title = {'Title': meta.pop('Title')} if 'Title' in meta else {}    # Get the 'Title' key-value pair and append at the first index
        meta = {**title, **meta}    # Inserts 'Title' at index 0
        metadata = ''
        for k, v in meta.items():
            metadata += f"{k}: {v}\n\n"        
        return answer, metadata
    
    else:
        # initialize the vector store object
        vectorstore = PineconeVectorStore(
            index_name='spenaic-papers',
            embedding=embeddings,
            pinecone_api_key=os.getenv('PINECONE_APIKEY_CONTENT')
        ).as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold':0.8, 'k':5}
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=ConversationBufferWindowMemory(k=3, memory_key='chat_history', output_key='answer', return_messages=True),
            retriever=vectorstore,
            condense_question_prompt=prompt,
            return_source_documents=True,
            response_if_no_docs_found="I'm sorry, I couldn't find any relevant documents."
        )
        response = chain.invoke({'question': query})
        answer = response['answer']
        for phrase in ['I don\'t know','I do not know','provided context','I cannot find','I can\'t find','I\'m just an Energy Industry Chatbot']:
            if phrase in answer:
                return answer, ''

        # Merging all retrieved metadata into one, while retaining only the unique values
        merged_metadata = defaultdict(list)     # creates a dictionary with default value as a list
        for doc in response['source_documents']:    # iterate over the source documents
            for k, v in doc.metadata.items():   # iterate over the key-value pair in the metadata of current document
                merged_metadata[k].append(v)
        meta = {k: list(set(v))[0] for k, v in merged_metadata.items()}    # removes duplicates from the metadata
        title = {'Title': meta.pop('Title')} if 'Title' in meta else {}    # Get the 'Title' key-value pair and append at the first index
        meta = {**title, **meta}    # Inserts 'Title' at index 0
        metadata = ''
        for k, v in meta.items():
            metadata += f"{k}: {v}\n\n"
        return answer, metadata        


def save_conversation_history():
    conversation_history = {
        'messages': st.session_state.get('messages', [{"role":"assistant", "content":"Hello, there.\n How can I assist you?"}])
    }
    with open('conversation_history.json', 'w') as f:
        json.dump(conversation_history, f)


def load_conversation_history():
    try:
        with open('conversation_history.json', 'r') as f:
            conversation_history = json.load(f)
            st.session_state.messages = conversation_history.get('messages', [{"role":"assistant", "content":"Hello, there.\n How can I assist you?"}])
    except FileNotFoundError:
        st.session_state.messages = [{"role":"assistant", "content":"Hello, there.\n How can I assist you?"}]




# Replace DATE with the specific date folder containing the text files to be vectorized
DATE = 'August 2022'

if __name__ == '__main__':
    # YOU SHOULD RUN THIS SCRIPT ONLY WHEN YOU HAVE NEWER TEXT FILES THAT HASN'T BEEN EMBEDDED AND STORED IN PINECONE
    list_of_docs = combine_docs(os.path.join(os.getcwd(), 'files', DATE))
    doc_chunks = split_to_chunks(list_of_docs)
    vectorize_doc_chunks(doc_chunks)