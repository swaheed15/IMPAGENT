import os
import logging
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import PubMedAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType 
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from scholarly import scholarly
from dotenv import load_dotenv 
import requests
from xml.etree import ElementTree

# Load environment variables
load_dotenv()

# Retrieve API keys from environment or secrets
pubmed_api_key = st.secrets["PUBMED_API_KEY"]
from urllib.parse import quote
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def build_esearch_url(query: str):
    return f"{NCBI_BASE_URL}esearch.fcgi?db=pubmed&term={query}&api_key={pubmed_api_key}&retmode=xml"

def fetch_pubmed_details(pmids: list):
    """Fetch detailed information for a list of PubMed IDs using the esummary endpoint."""
    esummary_url = f"{NCBI_BASE_URL}esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),  # Join PMIDs into a comma-separated string
        "api_key": pubmed_api_key,
        "retmode": "xml",  # Request XML response
    }
    response = requests.get(esummary_url, params=params)

    if response.status_code == 200:
        try:
            root = ElementTree.fromstring(response.text)  # Parse the XML response
            articles = []
            for docsum in root.findall(".//DocSum"):
                pmid = docsum.find(".//Id").text
                title = docsum.find(".//Item[@Name='Title']").text
                source = docsum.find(".//Item[@Name='Source']").text
                pubdate = docsum.find(".//Item[@Name='PubDate']").text
                abstract = docsum.find(".//Item[@Name='AbstractText']")
                abstract = abstract.text if abstract is not None else "No abstract available"
                authors = [
                    author.text for author in docsum.findall(".//Item[@Name='Author']")
                ]
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "source": source,
                    "pubdate": pubdate,
                    "abstract": abstract,
                    "authors": authors,
                })
            return articles
        except ElementTree.ParseError:
            raise ValueError("Failed to parse XML response from PubMed.")
    else:
        raise ValueError(f"Failed to fetch PubMed results. HTTP Status Code: {response.status_code}")

def parse_pubmed_xml(xml_data):
    """Parse XML data from PubMed and extract PMIDs."""
    root = ElementTree.fromstring(xml_data)
    pmids = []
    for id_element in root.findall(".//Id"):
        pmids.append(id_element.text)  # Extract PMIDs
    return pmids

def pubmed_tool_func(query: str, top_k_results=5, callback=None):
    try:
        result = fetch_pubmed_details(query)
        pmids = parse_pubmed_xml(result)[:top_k_results]  # Limit results
        articles = fetch_pubmed_details(pmids)
        if callback:
            callback(articles)
        return articles
    except Exception as e:
        logging.error(f"Error fetching Google Scholar results: {e}")
    return []

pubmed_tool = Tool(
    name="PubMedQuery",
    description="Search PubMed for medical and scientific literature using NCBI E-utilities.",
    func=pubmed_tool_func,
    is_single_input=True
)

def google_scholar_query(query, num_results=10):
    search_results = scholarly.search_pubs(query)
    results = []
    try:
        for _ in range(num_results):
            result = next(search_results)
            results.append({
                "title": result.get("bib",{}).get("title", " No title available"),
                "author": result.get("bib",{}).get("author", " No authors available"),
                "year": result.get("bib",{}).get("year", " No year available"),
                "abstract": result.get("bib",{}).get("abstract", " No abstract available"),
                "url": result.get("bib",{}).get("url", " No URL available"),
            })
    except StopIteration: 
        pass
    except Exception as e:
        logging.error(f"Error fetching Google Scholar results: {e}")
    return results

google_scholar_tool = Tool(
    name="GoogleScholarQuery",
    description="Search Google Scholar for academic papers.",
    func=google_scholar_query,
    is_single_input=True
)

groq_api_key = st.sidebar.text_input("Please Enter your Groq API key:", type="password")
if not groq_api_key:
    st.warning("Please enter your Groq API key to use the summarization feature.")
    st.stop() 
from langchain_groq import ChatGroq 
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it",
    streaming=True
)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def summarize_text(text: str, max_length: int = 250) -> str:
    """Summarize the given text to fit within the max_length."""
    if len(text) > max_length:
        prompt_template = PromptTemplate(
            input_variables=["text", "max_length"],
            template="Summarize the following text in {max_length} characters:\n\n{text}"
        )
        summarization_chain = LLMChain(llm=llm, prompt=prompt_template)
        summary = summarization_chain.run({"text": text, "max_length": max_length})
        return summary
    return text
# Streamlit app 
st.title("Research Agent")
st.write("This agent helps you search PubMed and Google Scholar")

top_k_results = st.slider("Top Results:", 1, 10, 5)
doc_content_chars_max = st.slider("Max Characters:", 100, 500, 250)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "Assistant", "content": "Hi, I am your research assistant. How can I help you?"}]

with st.container():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Search: "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Logging the settings
    logging.info(f"Top K Results: {top_k_results}, Max Characters: {doc_content_chars_max}")

    # Define tools
    tools = [pubmed_tool, google_scholar_tool]

    # Initialize the search agent
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        if isinstance(response, list) and all(isinstance(article, dict) for article in response):
            summarized_articles = [
                {
                    "abstract_summary": summarize_text(article.get('abstract', 'No Abstract available'), max_length=500),  # Summarize abstract
                    "full_text_summary": summarize_text(article.get('full_text', 'No Full text available'), max_length=1000)  # Summarize full text if available
                }
                for article in response
            ]
            for article, summary in zip(response, summarized_articles):
                st.write(f"**Title:** {article.get('title', 'No title available')}")
                st.write(f"**Abstract (Summarized):** {summary.get('abstract_summary', 'No Abstract available')}") 
                st.write(f"**Full Text (Summarized):** {summary.get('full_text_summary', 'No Full text available')}") 
                st.write(f"**PubMed ID:** {article.get('pmid', 'N/A')}")
                st.write(f"**Authors:** {', '.join(article.get('authors', ['No authors available']))}")
                st.write(f"**Year:** {article.get('pubdate', 'N/A')}")
                st.write("---") 
            else:
                st.write(response)