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
# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

memory = ConversationBufferWindowMemory(k=3, return_messages=True)

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
                mydoc = Document(page_content=value, metadata={'Authors':doc.metadata['Authors'], 'Presentation date':doc.metadata['Publication Date'], 'ref link':doc.metadata['Reference Link']})
                docs.append(mydoc)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_APIKEY_TITLE'))
    vectorstore.add_documents(docs)


@st.cache_data(show_spinner=False)
def find_similar_papers(paper_title:str, k:int=5, date:str=None, index_name:str='paper-title') -> list:
    '''`date`: `'month YYYY'`\n
        Uses similarity search to retrieve 5 most-similar papers according to a given paper title\n
        Where date is given, metadata filtering is applied to further narrow down the search
    '''
    papers = list(); ref_links = list()
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_APIKEY_TITLE'))
    if date != None:
        docs = vectorstore.similarity_search(paper_title, k=k, filter={'date': {"$eq": f"{date}"}})
        if docs == []:
            return docs
        for doc in docs:
            papers.append(f"{doc.page_content}  (Date: {doc.metadata['Presentation date']})")
            ref_links.append(doc.metadata['ref link'])
    else:
        docs = vectorstore.similarity_search(paper_title, k=k)
        if docs == []:
            return docs
        for doc in docs:
            papers.append(f"{doc.page_content}  (Date: {doc.metadata['Presentation date']})")
            ref_links.append(doc.metadata['ref link'])
    return papers, ref_links


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
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.5)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    doc_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return doc_chain.run(doc), metadata


@st.cache_data(show_spinner=False)
def get_response(query:str):
    '''Generates a response to User query, while also providing a list of similar papers'''
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    llm = GoogleGenerativeAI(google_api_key=os.environ['GOOGLE_API_KEY'], model='gemini-pro', temperature=0.7)
    # llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # initialize the vector store object
    vectorstore = PineconeVectorStore(
        index_name='spenaic-papers',
        embedding=embeddings,
        pinecone_api_key=os.getenv('PINECONE_APIKEY_CONTENT')
    ).as_retriever(
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
      ("system", "Answer the human's questions based on the given context. However, if you cannot find an answer based on the context, simply say - 'My apologies! As an Energy Industry Chatbot, I\'m only able to respond to queries related to the Energy Industry':\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{input}"),
    ])
    # create a runnable that, when invoked, appends retrieved List[Docs] to prompt and passes it on to the LLM as context for generating response to user_input
    context_to_response_runnable = create_stuff_documents_chain(llm, elicit_response_prompt)

    # chains up two runnables to yield the final output that would include human_query, chat_history, context and answer
    retrieval_chain_runnable =  create_retrieval_chain(doc_retriever_runnable, context_to_response_runnable)     # chains up two runnables to yield the final output that would include human_query, chat_history, context and answer

    response = retrieval_chain_runnable.invoke({
        "chat_history": memory.load_memory_variables({})['history'],
        "input": query
    })
    # since the memory is a buffer window, we save the context of the current conversation. However, if it were in a conversation chain, the memory would be updated automatically based on its window size after each conversation
    memory.save_context({"input": f"{response['input']}"}, {"output": f"{response['answer']}"})

    for phrase in ['I don\'t know','I do not know','not in provided context',"My apologies! As an Energy Industry Chatbot, I'm only able to respond to queries related to the Energy Industry"]:
        if phrase in response['answer']:
            return response['answer'], ''
        
    papers = [docs.metadata['Title'] for docs in response['context']]
    most_frequent_paper = Counter(papers).most_common(1)[0][0]
    paper_titles, links = find_similar_papers(most_frequent_paper, k=4)

    return response['answer'], list(zip(paper_titles, links))


def save_conversation_history():
    chat_history = {
        'messages': st.session_state.get('messages', [{"role":"assistant", "content":"Hello, there.\n How can I assist you?"}])
    }
    with open('chat_history.json', 'w') as f:
        json.dump(chat_history, f)


def load_conversation_history():
    try:
        with open('chat_history.json', 'r') as f:
            chat_history = json.load(f)
            st.session_state.messages = chat_history.get('messages', [{"role":"assistant", "content":"Hello, there.\n How can I assist you?"}])
    except FileNotFoundError:
        st.session_state.messages = [{"role":"assistant", "content":"Hello, there.\n How may I assist you?"}]



# Replace DATE with the specific date folder containing the text files to be vectorized
DATE = 'August 2022'

if __name__ == '__main__':
    # YOU SHOULD RUN THIS SCRIPT ONLY WHEN YOU HAVE NEWER TEXT FILES THAT HASN'T BEEN EMBEDDED OR/AND STORED IN PINECONE
    list_of_docs = combine_docs(os.path.join(os.getcwd(), 'files', DATE))
    doc_chunks = split_to_chunks(list_of_docs)
    vectorize_doc_chunks(doc_chunks)
    vectorize_paper_titles(os.path.join(os.getcwd(), 'files', DATE))