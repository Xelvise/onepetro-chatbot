# RAG-based App for querying Research papers

[OnePetro](https://onepetro.org/) is an online library containing research papers from several journals to conference proceedings. This project aims to develop a Chatbot that helps Researchers swiftly and more efficiently access the right information, as well as citations, to their searches.

Integrated in the app is a RAG-based interactive Chatbot serving as a retrieval agent between a Researcher and the chunks of resources in [OnePetro](https://onepetro.org/).

Under the hood, there is a Web automation script that scrapes Conference papers from OnePetro's Catalog and stores them locally (in .txt). These text files are further splitted into chunks, embedded and put in a Vectorstore for subsequent retrieval.

### App Functionalities include:

- **Insightful Summary**: User-selected papers are summarised in a way that reveals the key points that may be insightful to the Researcher.

- **Interactive Chat**:  The whole idea here is - In a case where the Paper overview isn't explanatory enough to satisfy the Researcher's curiosity, the Chat feature comes in handy. 
Furthermore, Researchers can throw their questions at the Bot which will return a Query-specific answer... so long as it pertains to the resources available in the OnePetro's Catalog. Moreover, the Bot also gives contextual responses by using previous conversations to infer the next response.

- **Citations**: Attached to each response of the Chatbot are citations of related papers with their hyperlinks that the Researcher may find helpful. You can think of this like a Recommendation system, based on User preference (i.e, selected paper).

It's worth noting that the Chatbot isn't a generic one for casual chats. Once spotted, the chatbot gives a reprimand, saying - `My apologies! As an Energy Industry Chatbot, I'm only able to respond to queries related to the Energy Industry.`

No citation whatsoever is given to a generic query. This is to ensure that the Chatbot is used for its intended purpose only.

### Local Installation:

1. Clone the Repository
    ```
    git clone https://github.com/Xelvise/onepetro-chatbot.git
    ```
2. Create and activate a Virtual Environment (Optional, but recommended)
    ```
    cd onepetro-chatbot
    conda create -p <Environment_Name> python==<python version> -y
    conda activate <Environment_Name>/
    ```
3. While inside onepetro-chatbot directory, Install dependencies
    ```
    pip install -r requirements.txt
    ```
4. Run and explore the app in your browser
    ```
    streamlit run Explore_articles.py
    ```
5. To scrape even more papers, and have them stored locally, run the web automation script:
    ```
    python scraper.py
    ```
    Prior to execution, ensure to add the path to your browser's `user_data_dir` at the specified location in the script. The path can be found by entering `chrome://version/` in the address bar of your Chrome browser.

6. To have the newly-scraped papers (in .txt) added to the Vectorstore:
    ```
    python utils.py
    ```
    This script will update the Vectorstore with the new papers, thereby making it accessible for subsequent queries.

### App Demo:
![Paper Overview](https://github.com/Xelvise/onepetro-chatbot/blob/main/img/explore_section.png?raw=true "Title")

![Chatbot showing citations](https://github.com/Xelvise/onepetro-chatbot/blob/main/img/chat_section.png?raw=true "Title")

![Guardrails to prevent generic queries](https://github.com/Xelvise/onepetro-chatbot/blob/main/img/chat2_section.png?raw=true "Title")


### Tools/Frameworks and Technologies used:
1. [Streamlit](https://streamlit.io/) for App UI
2. Google Gemini for LLM
3. Selenium and Beautiful-Soup for Web automation, scraping and parsing text
4. Pinecone for Vectorstore
5. Langchain for creating connection btw LLM, Retrieved docs, Prompts and Conversations