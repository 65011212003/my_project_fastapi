"""
FastAPI Application for Rice Disease Analysis

This application fetches articles from PubMed based on a query,
processes the abstracts, performs clustering and topic modeling,
extracts entities, and provides API endpoints to interact with the data.
"""

import os
import time
import logging
import re
import string
import subprocess
import sys
from typing import List, Optional, Dict

import matplotlib
# Set the backend to 'Agg' to avoid GUI warnings/errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Bio import Entrez
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim import corpora, models

from fastapi.responses import FileResponse
from fastapi import Query
import json
from datetime import datetime, timedelta
import io
import aiohttp

import random

from googletrans import Translator

# -----------------------------------
# Setup Logging
# -----------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -----------------------------------
# Download NLTK and spaCy Data
# -----------------------------------
def download_nltk_data():
    nltk_packages = ['punkt', 'stopwords', 'wordnet']
    for package in nltk_packages:
        nltk.download(package)

def download_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        logging.info("Downloading 'en_core_web_sm' for spaCy.")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# -----------------------------------
# Initialize NLP Tools
# -----------------------------------
download_nltk_data()
download_spacy_model()
nlp = spacy.load("en_core_web_sm")

# -----------------------------------
# Configure Entrez
# -----------------------------------
Entrez.email = "65011212003@msu.ac.th"
Entrez.api_key = "250b38811eabf58300fe369fa32371342308"
# -----------------------------------
# Initialize FastAPI
# -----------------------------------
app = FastAPI(title="Rice Disease Analysis API")

# -----------------------------------
# Configure CORS
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# -----------------------------------
# Pydantic Models
# -----------------------------------
class PubMedQuery(BaseModel):
    query: str
    max_results: Optional[int] = 10000

class AnalysisResponse(BaseModel):
    message: str
    cluster_explanations: Optional[List[dict]] = None

class ClusterInfo(BaseModel):
    cluster_number: int
    article_count: int
    top_terms: List[str]

class Article(BaseModel):
    PMID: str
    Title: str
    Abstract: str
    Processed_Abstract: str
    Cluster: int
    Entities: dict
    Dominant_Topic: Optional[int]

class SearchQuery(BaseModel):
    keyword: str
    field: str = "Title"  # Can be "Title", "Abstract", or "All"

class StatisticsResponse(BaseModel):
    disease_counts: dict
    yearly_trends: dict
    top_keywords: List[dict]

class TopicExplanation(BaseModel):
    topic_id: int
    main_focus: str
    simple_explanation: str
    key_terms: List[str]
    relevance_score: float

class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[Article]

class NewsArticle(BaseModel):
    title: str
    description: Optional[str]
    url: str
    urlToImage: Optional[str]
    publishedAt: str
    source: dict

class ResearchTrend(BaseModel):
    year: str
    cases: int

class RelatedResearch(BaseModel):
    title: str
    authors: List[str]
    year: str

class ResearchLocation(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    diseases: List[str]
    severity: float  # 0-10
    researchCount: int
    trends: List[ResearchTrend]
    recommendations: List[str]
    relatedResearch: List[RelatedResearch]

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class BulkTranslationRequest(BaseModel):
    texts: List[str]
    target_language: str

# -----------------------------------
# Utility Functions
# -----------------------------------
def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by lowercasing, removing numbers, punctuation,
    tokenizing, removing stopwords, and lemmatizing.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    preprocessed = ' '.join(tokens)

    if not preprocessed:
        logging.warning("Preprocessed abstract is empty.")

    return preprocessed

def extract_entities_simple(text: str) -> dict:
    """
    Extract entities like diseases, symptoms, treatments from text using simple heuristics.
    """
    doc = nlp(text)
    entities = {"Disease": [], "Treatment": [], "Symptom": []}
    for chunk in doc.noun_chunks:
        # Simple heuristic rules
        if any(word in chunk.text.lower() for word in ["disease", "virus", "fungus"]):
            entities["Disease"].append(chunk.text)
        if any(word in chunk.text.lower() for word in ["treatment", "method", "management"]):
            entities["Treatment"].append(chunk.text)
        if any(word in chunk.text.lower() for word in ["effect", "impact", "symptom"]):
            entities["Symptom"].append(chunk.text)
    return entities

def search_pubmed(query: str, max_results: int = 1000) -> List[str]:
    """
    Search PubMed for the given query and return a list of PubMed IDs (PMIDs).
    """
    handle = Entrez.esearch(
        db="pubmed",
        sort="relevance",
        retmax=max_results,
        retmode="xml",
        term=query
    )
    results = Entrez.read(handle)
    handle.close()
    return results.get("IdList", [])

def fetch_details(id_list: List[str]) -> List[dict]:
    """
    Fetch details for each PMID in id_list using XML parsing.
    Returns a list of dictionaries with PMID, Title, and Abstract.
    """
    records = []
    for start in range(0, len(id_list), 200):
        end = min(len(id_list), start + 200)
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list[start:end],
            rettype="xml",
            retmode="xml"
        )
        data = Entrez.read(fetch_handle)
        fetch_handle.close()
        time.sleep(0.3)  # To respect NCBI rate limits

        for article in data.get('PubmedArticle', []):
            medline_citation = article.get('MedlineCitation', {})
            pmid_element = medline_citation.get('PMID', '')

            if isinstance(pmid_element, dict):
                pmid = pmid_element.get('#text', '')
            else:
                pmid = str(pmid_element)

            article_info = medline_citation.get('Article', {})
            title = article_info.get('ArticleTitle', '')

            abstract = ''
            abstract_texts = article_info.get('Abstract', {}).get('AbstractText', [])
            if isinstance(abstract_texts, list):
                abstract = ' '.join([str(text) for text in abstract_texts])
            elif isinstance(abstract_texts, str):
                abstract = abstract_texts

            if abstract.strip():
                records.append({
                    "PMID": pmid,
                    "Title": title,
                    "Abstract": abstract.strip()
                })
            else:
                logging.debug(f"PMID {pmid} has an empty abstract and will be skipped.")

    return records

def vectorize_text(processed_texts: List[str], max_features: int = 5000):
    """
    Vectorize the text using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    return X, vectorizer, feature_names

def determine_optimal_clusters(X, k_min=2, k_max=15):
    """
    Determine the optimal number of clusters using Elbow Method and Silhouette Score.
    Returns the optimal k value along with metrics for visualization.
    """
    sse = []
    silhouette_scores = []
    k_range = range(k_min, k_max)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        sse.append(km.inertia_)
        silhouette_avg = silhouette_score(X, km.labels_)
        silhouette_scores.append(silhouette_avg)
        logging.info(f"Cluster {k}, SSE: {km.inertia_}, Silhouette Score: {silhouette_avg:.4f}")
    
    # Find optimal k using silhouette score
    best_k = max(
        zip(k_range, silhouette_scores),
        key=lambda x: x[1]
    )[0]
    
    logging.info(f"Selected optimal number of clusters: {best_k}")
    return k_range, sse, silhouette_scores, best_k

def perform_kmeans(X, n_clusters=5):
    """
    Perform K-Means clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters

def perform_lda(tokenized_docs: List[List[str]], num_topics: int = 5, passes: int = 10):
    """
    Perform LDA topic modeling.
    """
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    logging.info("LDA model training complete.")
    return lda_model, corpus, dictionary

def get_dominant_topic(ldamodel, corpus):
    """
    Returns the dominant topic for each text in corpus.
    """
    dominant_topics = []
    for row in ldamodel.get_document_topics(corpus):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        if row:
            dominant_topic = row[0][0]
        else:
            dominant_topic = None
        dominant_topics.append(dominant_topic)
    return dominant_topics

# -----------------------------------
# API Endpoints
# -----------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
def analyze_data(query: PubMedQuery, background_tasks: BackgroundTasks):
    """
    Initiate the analysis process with the given PubMed query.
    The processing is done in the background.
    """
    # Get cluster terms from existing data if available
    cluster_terms = []
    try:
        df = pd.read_csv("rice_disease_analysis/rice_disease_pubmed_data.csv")
        processed_abstracts = df['Processed_Abstract'].tolist()
        X, vectorizer, feature_names = vectorize_text(processed_abstracts)
        optimal_k = df['Cluster'].nunique()
        
        for cluster in range(optimal_k):
            cluster_features = X[df['Cluster'] == cluster].toarray()
            if cluster_features.size > 0:
                avg_features = np.mean(cluster_features, axis=0)
                top_terms = [feature_names[i] for i in np.argsort(avg_features)[-10:]]
                cluster_terms.append(top_terms)
    except:
        pass
    
    # Get human-friendly explanations
    explanations = get_human_friendly_cluster_explanations(cluster_terms) if cluster_terms else None
    
    # Start the background task
    background_tasks.add_task(process_data, query.query, query.max_results)
    
    return {
        "message": "Analysis started. Please check the results after completion.",
        "cluster_explanations": explanations
    }

def process_data(query: str, max_results: int):
    logging.info("Starting data processing...")

    # Define output directory early to ensure it's available for all outputs
    output_dir = "rice_disease_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data Collection
    pmid_list = search_pubmed(query, max_results=max_results)
    logging.info(f"Found {len(pmid_list)} articles.")

    articles = fetch_details(pmid_list)
    logging.info(f"Collected {len(articles)} articles with abstracts.")

    if not articles:
        logging.error("No articles fetched. Exiting process.")
        return

    df = pd.DataFrame(articles, columns=["PMID", "Title", "Abstract"])
    initial_count = len(df)

    # 2. Data Preprocessing
    df['Processed_Abstract'] = df['Abstract'].apply(preprocess_text)
    df.drop_duplicates(subset=['Abstract'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    removed_duplicates = initial_count - len(df)
    logging.info(f"Removed {removed_duplicates} duplicate articles based on abstracts.")

    # Remove empty processed abstracts
    preprocessed_initial_count = len(df)
    df = df[df['Processed_Abstract'].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    removed_empty_preprocessed = preprocessed_initial_count - len(df)
    if removed_empty_preprocessed > 0:
        logging.info(f"Removed {removed_empty_preprocessed} articles with empty processed abstracts.")
    logging.info(f"Articles remaining after preprocessing: {len(df)}")

    if len(df) == 0:
        logging.error("No articles left after preprocessing. Exiting process.")
        return

    # 3. Vectorization
    X, vectorizer, feature_names = vectorize_text(df['Processed_Abstract'])
    logging.info("Vectorization successful.")

    # 4. Determine Optimal Number of Clusters
    k_range, sse, silhouette_scores, optimal_k = determine_optimal_clusters(X)
    
    # Plot Elbow and Silhouette Scores
    plt.figure(figsize=(14,6))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Distances (SSE)")
    plt.title("Elbow Method For Optimal k")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', color='orange')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores For Various k")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "clustering_plots.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Clustering plots saved to {plot_path}.")

    logging.info(f"Clustering into {optimal_k} clusters using K-Means.")
    kmeans, clusters = perform_kmeans(X, n_clusters=optimal_k)
    df['Cluster'] = clusters

    # 5. Entity Extraction
    df['Entities'] = df['Processed_Abstract'].apply(extract_entities_simple)

    # 6. Topic Modeling with LDA
    tokenized_docs = df['Processed_Abstract'].tolist()
    tokenized_docs = [doc.split() for doc in tokenized_docs]
    lda_model, corpus, dictionary = perform_lda(tokenized_docs, num_topics=optimal_k)
    df['Dominant_Topic'] = get_dominant_topic(lda_model, corpus)

    # 7. Save Results
    data_csv_path = os.path.join(output_dir, "rice_disease_pubmed_data.csv")
    df.to_csv(data_csv_path, index=False)
    logging.info(f"Data saved to {data_csv_path}.")

    # 8. Save LDA Topics
    lda_topics = lda_model.print_topics(-1)
    lda_topics_path = os.path.join(output_dir, "lda_topics.txt")
    with open(lda_topics_path, "w") as f:
        for idx, topic in lda_topics:
            f.write(f"Topic {idx}: {topic}\n")
    logging.info(f"LDA topics saved to {lda_topics_path}.")

@app.get("/status", response_model=str)
def get_status():
    """
    Check if the API is running.
    """
    return "Rice Disease Analysis API is running."

@app.get("/clusters", response_model=List[ClusterInfo])
def get_clusters():
    """
    Retrieve information about each cluster, including the number of articles and top terms.
    """
    data_path = "rice_disease_analysis/rice_disease_pubmed_data.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data not found. Please run the analysis first.")

    df = pd.read_csv(data_path)
    processed_abstracts = df['Processed_Abstract'].tolist()
    X, vectorizer, feature_names = vectorize_text(processed_abstracts)
    optimal_k = df['Cluster'].nunique()

    results = []
    for cluster in range(optimal_k):
        cluster_features = X[df['Cluster'] == cluster].toarray()
        if cluster_features.size == 0:
            top_features = []
        else:
            avg_features = np.mean(cluster_features, axis=0)
            top_features = [feature_names[i] for i in np.argsort(avg_features)[-10:]]
        results.append(ClusterInfo(
            cluster_number=cluster,
            article_count=int(df['Cluster'].value_counts().get(cluster, 0)),
            top_terms=top_features
        ))
    return results

@app.get("/articles/{cluster_id}", response_model=List[Article])
def get_articles_by_cluster(cluster_id: int):
    """
    Retrieve articles belonging to a specific cluster.
    """
    data_path = "rice_disease_analysis/rice_disease_pubmed_data.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data not found. Please run the analysis first.")

    df = pd.read_csv(data_path)
    cluster_df = df[df['Cluster'] == cluster_id]

    if cluster_df.empty:
        raise HTTPException(status_code=404, detail=f"No articles found for cluster {cluster_id}.")

    articles = []
    for _, row in cluster_df.iterrows():
        # การแปลง string ที่เป็น dict กลับเป็น dict จริงๆ
        entities = row['Entities']
        if isinstance(entities, str):
            try:
                entities = eval(entities)
            except:
                entities = {}
        articles.append(Article(
            PMID=row['PMID'],
            Title=row['Title'],
            Abstract=row['Abstract'],
            Processed_Abstract=row['Processed_Abstract'],
            Cluster=int(row['Cluster']),
            Entities=entities,
            Dominant_Topic=int(row['Dominant_Topic']) if not pd.isna(row['Dominant_Topic']) else None
        ))
    return articles

@app.get("/topics", response_model=List[str])
def get_lda_topics():
    """
    Retrieve the LDA topics.
    """
    topics_path = "rice_disease_analysis/lda_topics.txt"
    if not os.path.exists(topics_path):
        raise HTTPException(status_code=404, detail="LDA topics not found. Please run the analysis first.")

    with open(topics_path, "r") as f:
        topics = f.readlines()
    return [topic.strip() for topic in topics]

@app.get("/csv-data")
def get_csv_data():
    """
    Retrieve CSV data for display in the web interface.
    """
    data_path = "rice_disease_analysis/rice_disease_pubmed_data.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data not found. Please run the analysis first.")

    df = pd.read_csv(data_path)
    
    # Get headers and first 10 rows for display
    headers = df.columns.tolist()
    rows = df.head(10).values.tolist()
    
    return {
        "headers": headers,
        "rows": rows
    }

@app.get("/export/{format}")
def export_data(format: str):
    """
    Export analysis results in CSV or JSON format.
    """
    data_path = "rice_disease_analysis/rice_disease_pubmed_data.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data not found. Please run the analysis first.")

    df = pd.read_csv(data_path)
    
    if format.lower() == "csv":
        # Create a buffer to store the CSV data
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Create filename with timestamp
        filename = f"rice_disease_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return FileResponse(
            path=data_path,
            filename=filename,
            media_type="text/csv"
        )
    
    elif format.lower() == "json":
        # Convert DataFrame to JSON
        json_data = df.to_json(orient="records")
        filename = f"rice_disease_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save JSON to file
        json_path = os.path.join("rice_disease_analysis", filename)
        with open(json_path, "w") as f:
            f.write(json_data)
            
        return FileResponse(
            path=json_path,
            filename=filename,
            media_type="application/json"
        )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'.")

@app.post("/search", response_model=List[Article])
def search_articles(query: SearchQuery):
    """
    Search articles by keyword in title or abstract.
    """
    data_path = "rice_disease_analysis/rice_disease_pubmed_data.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data not found. Please run the analysis first.")

    try:
        df = pd.read_csv(data_path)
        logging.info(f"Searching for keyword '{query.keyword}' in field '{query.field}'")
        
        # Convert keyword to lowercase for case-insensitive search
        keyword = query.keyword.lower()
        
        if query.field == "Title":
            mask = df['Title'].str.lower().str.contains(keyword, na=False)
        elif query.field == "Abstract":
            mask = df['Abstract'].str.lower().str.contains(keyword, na=False)
        else:  # All
            mask = (df['Title'].str.lower().str.contains(keyword, na=False) | 
                   df['Abstract'].str.lower().str.contains(keyword, na=False))
        
        filtered_df = df[mask]
        logging.info(f"Found {len(filtered_df)} matching articles")
        
        if filtered_df.empty:
            return []
        
        articles = []
        for _, row in filtered_df.iterrows():
            try:
                entities = row['Entities']
                if isinstance(entities, str):
                    entities = eval(entities)
                
                # Convert PMID to string and handle NaN values
                pmid = str(row['PMID']) if pd.notna(row['PMID']) else ''
                
                # Handle potential NaN values in other fields
                article = Article(
                    PMID=pmid,
                    Title=str(row['Title']) if pd.notna(row['Title']) else '',
                    Abstract=str(row['Abstract']) if pd.notna(row['Abstract']) else '',
                    Processed_Abstract=str(row['Processed_Abstract']) if pd.notna(row['Processed_Abstract']) else '',
                    Cluster=int(row['Cluster']) if pd.notna(row['Cluster']) else 0,
                    Entities=entities if isinstance(entities, dict) else {},
                    Dominant_Topic=int(row['Dominant_Topic']) if pd.notna(row['Dominant_Topic']) else None
                )
                articles.append(article)
            except Exception as e:
                logging.error(f"Error processing article {pmid}: {str(e)}")
                continue
        
        return articles
        
    except Exception as e:
        logging.error(f"Error in search_articles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching articles: {str(e)}")

@app.get("/statistics", response_model=StatisticsResponse)
def get_statistics():
    """
    Get statistical information about the analyzed data.
    """
    data_path = "rice_disease_analysis/rice_disease_pubmed_data.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data not found. Please run the analysis first.")

    df = pd.read_csv(data_path)
    
    # Count diseases mentioned
    disease_counts = {}
    for _, row in df.iterrows():
        entities = row['Entities']
        if isinstance(entities, str):
            try:
                entities = eval(entities)
                if 'Disease' in entities:
                    for disease in entities['Disease']:
                        disease_counts[disease] = disease_counts.get(disease, 0) + 1
            except:
                continue
    
    # Get top 10 diseases
    disease_counts = dict(sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Extract years from titles or abstracts (assuming year is mentioned)
    years = []
    pattern = r'\b(19|20)\d{2}\b'
    for text in df['Abstract']:
        if isinstance(text, str):
            found_years = re.findall(pattern, text)
            years.extend(found_years)
    
    yearly_trends = {}
    for year in years:
        yearly_trends[year] = yearly_trends.get(year, 0) + 1
    
    # Get top keywords from processed abstracts
    words = []
    for text in df['Processed_Abstract']:
        if isinstance(text, str):
            words.extend(text.split())
    
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    top_keywords = [{"word": k, "count": v} 
                   for k, v in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]]
    
    return StatisticsResponse(
        disease_counts=disease_counts,
        yearly_trends=yearly_trends,
        top_keywords=top_keywords
    )

@app.get("/farmer-topics", response_model=List[TopicExplanation])
def get_farmer_friendly_topics():
    """
    Get farmer-friendly explanations of the LDA topics.
    """
    topics_path = "rice_disease_analysis/lda_topics.txt"
    if not os.path.exists(topics_path):
        raise HTTPException(status_code=404, detail="LDA topics not found. Please run the analysis first.")

    with open(topics_path, "r") as f:
        topics = f.readlines()

    farmer_friendly_topics = []
    
    # Create farmer-friendly explanations for each topic
    topic_explanations = {
        0: {
            "main_focus": "การป้องกันโรคข้าว",
            "simple_explanation": "เกี่ยวกับวิธีการป้องกันและต่อต้านโรคในนาข้าว รวมถึงการจัดการพันธุ์ข้าวที่ทนทาน",
            "key_terms": ["การป้องกัน", "ความต้านทาน", "พันธุ์ข้าว", "การจัดการโรค"]
        },
        1: {
            "main_focus": "โรคข้าวและเชื้อก่อโรค",
            "simple_explanation": "เกี่ยวกับชนิดของโรคข้าวและเชื้อที่ทำให้เกิดโรค รวมถึงการระบาดในพื้นที่ต่างๆ",
            "key_terms": ["โรคข้าว", "เชื้อโรค", "การระบาด", "อาการ"]
        },
        2: {
            "main_focus": "การจัดการนาข้าว",
            "simple_explanation": "วิธีการดูแลนาข้าวให้แข็งแรง ลดการเกิดโรค และการจัดการน้ำและปุ๋ย",
            "key_terms": ["การจัดการ", "การดูแล", "การเพาะปลูก", "สภาพแวดล้อม"]
        },
        3: {
            "main_focus": "การตรวจสอบโรค",
            "simple_explanation": "วิธีสังเกตและตรวจหาโรคในนาข้าว รวมถึงการวินิจฉัยอาการเบื้องต้น",
            "key_terms": ["การตรวจสอบ", "อาการของโรค", "การวินิจฉัย", "การสังเกต"]
        },
        4: {
            "main_focus": "การวิจัยและพัฒนา",
            "simple_explanation": "ผลการศึกษาวิจัยใหม่ๆ เกี่ยวกับโรคข้าวและวิธีการป้องกันที่ได้ผล",
            "key_terms": ["การวิจัย", "การพัฒนา", "นวัตกรรม", "เทคโนโลยี"]
        }
    }

    for idx, topic in enumerate(topics):
        # Extract weights using regex to handle different formats
        try:
            # Remove "Topic X:" prefix if present
            topic_text = topic.split(':', 1)[-1].strip()
            # Extract all numbers before "*" symbols
            weights = []
            terms = topic_text.split('+')
            for term in terms:
                term = term.strip()
                if '*' in term:
                    weight_str = term.split('*')[0].strip()
                    try:
                        weight = float(weight_str)
                        weights.append(weight)
                    except ValueError:
                        continue
            
            # Calculate relevance score
            relevance_score = sum(weights) * 100 if weights else 50  # Default to 50 if no weights found
        except Exception as e:
            logging.error(f"Error parsing topic weights: {e}")
            relevance_score = 50  # Default score if parsing fails
        
        explanation = topic_explanations.get(idx, {
            "main_focus": "หัวข้ออื่นๆ",
            "simple_explanation": "ข้อมูลเพิ่มเติมเกี่ยวกับการดูแลและจัดการนาข้าว",
            "key_terms": ["การดูแล", "การจัดการ", "ข้อมูลเพิ่มเติม"]
        })

        farmer_friendly_topics.append(TopicExplanation(
            topic_id=idx,
            main_focus=explanation["main_focus"],
            simple_explanation=explanation["simple_explanation"],
            key_terms=explanation["key_terms"],
            relevance_score=round(relevance_score, 2)
        ))

    return farmer_friendly_topics

@app.get("/articles", response_model=PaginatedResponse)
async def get_articles(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
    try:
        df = pd.read_csv("rice_disease_analysis/rice_disease_pubmed_data.csv")
        
        # Calculate pagination values
        total_items = len(df)
        total_pages = (total_items + page_size - 1) // page_size
        
        # Ensure page is within valid range
        page = min(max(1, page), total_pages)
        
        # Get slice of data for current page
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        
        # Convert the slice of DataFrame to list of Articles
        articles = []
        for _, row in df.iloc[start_idx:end_idx].iterrows():
            article = Article(
                PMID=str(row['PMID']),
                Title=row['Title'],
                Abstract=row['Abstract'],
                Processed_Abstract=row['Processed_Abstract'],
                Cluster=int(row['Cluster']),
                Entities=eval(row['Entities']) if isinstance(row['Entities'], str) else row['Entities'],
                Dominant_Topic=int(row['Dominant_Topic']) if pd.notna(row['Dominant_Topic']) else None
            )
            articles.append(article)
        
        return PaginatedResponse(
            total=total_items,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            items=articles
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add News API configuration
NEWS_API_KEY = "fb075743fae14f89a5cc6bc75ff19009"
NEWS_API_URL = "https://newsapi.org/v2/everything"

@app.get("/news", response_model=List[NewsArticle])
async def get_rice_disease_news():
    """
    Fetch news articles about rice diseases from News API.
    """
    try:
        # Calculate date for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        params = {
            "q": "rice disease OR rice diseases OR rice pest OR rice farming",
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": NEWS_API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(NEWS_API_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("articles", [])
                    
                    # Validate and clean the articles data
                    cleaned_articles = []
                    for article in articles[:10]:  # Get top 10 most recent articles
                        try:
                            cleaned_article = {
                                "title": article.get("title", ""),
                                "description": article.get("description", ""),
                                "url": article.get("url", ""),
                                "urlToImage": article.get("urlToImage", ""),
                                "publishedAt": article.get("publishedAt", ""),
                                "source": article.get("source", {})
                            }
                            cleaned_articles.append(NewsArticle(**cleaned_article))
                        except Exception as e:
                            logging.error(f"Error processing article: {str(e)}")
                            continue
                    
                    return cleaned_articles
                else:
                    error_data = await response.json()
                    error_message = error_data.get("message", "Failed to fetch news articles")
                    raise HTTPException(
                        status_code=response.status,
                        detail=error_message
                    )
                    
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error connecting to News API: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching news: {str(e)}"
        )

def get_location_recommendations(diseases: List[str]) -> List[str]:
    """Generate context-aware recommendations based on diseases."""
    recommendations = {
        "โรคไหม้": [
            "ใช้พันธุ์ข้าวที่ต้านทานโรค",
            "หลีกเลี่ยงการใส่ปุ๋ยไนโตรเจนมากเกินไป",
            "กำจัดวัชพืชในนาข้าวและบริเวณใกล้เคียง",
            "ฉีดพ่นสารป้องกันกำจัดเชื้อราตามคำแนะนำ"
        ],
        "โรคขอบใบแห้ง": [
            "ใช้เมล็ดพันธุ์ที่ปลอดโรค",
            "ไม่ควรปลูกข้าวแน่นเกินไป",
            "ระบายน้ำในแปลงนาให้ทั่วถึง",
            "กำจัดหญ้าและพืชอาศัยของเชื้อโรค"
        ],
        "โรคใบจุดสีน้ำตาล": [
            "ปรับปรุงดินด้วยการใส่ปูนขาว",
            "ใช้ปุ๋ยโพแทสเซียมเพื่อเพิ่มความต้านทาน",
            "เก็บเกี่ยวในระยะที่เหมาะสม",
            "ทำความสะอาดแปลงนาหลังการเก็บเกี่ยว"
        ],
        "โรคกาบใบแห้ง": [
            "ลดความหนาแน่นของการปลูก",
            "ควบคุมระดับน้ำในนาให้เหมาะสม",
            "ฉีดพ่นสารป้องกันกำจัดเชื้อราในระยะกำเนิดช่อดอก",
            "ตากดินและไถกลบตอซัง"
        ]
    }
    
    result = []
    for disease in diseases:
        if disease in recommendations:
            result.extend(recommendations[disease])
    
    # Add general recommendations if list is empty or too short
    general_recommendations = [
        "ตรวจแปลงนาอย่างสม่ำเสมอเพื่อสังเกตอาการของโรค",
        "ปรึกษาเจ้าหน้าที่เกษตรในพื้นที่เมื่อพบปัญหา",
        "ทำความสะอาดเครื่องมือและอุปกรณ์การเกษตร",
        "วางแผนการปลูกให้เหมาะสมกับฤดูกาล"
    ]
    
    while len(result) < 4:
        result.append(random.choice(general_recommendations))
    
    return list(set(result))  # Remove duplicates

def generate_disease_trends(years: int = 5) -> List[ResearchTrend]:
    """Generate realistic disease trend data for the past few years."""
    current_year = datetime.now().year
    base_cases = random.randint(50, 200)
    trends = []
    
    for i in range(years):
        year = str(current_year - years + i + 1)
        # Add some random variation but maintain a trend
        variation = random.uniform(-0.3, 0.3)
        cases = int(base_cases * (1 + variation))
        base_cases = cases  # Use this as the base for next year
        trends.append(ResearchTrend(year=year, cases=cases))
    
    return trends

@app.get("/research-locations", response_model=List[ResearchLocation])
async def get_research_locations():
    """
    Get research location data for the map visualization.
    This endpoint aggregates research data by location and provides detailed information
    about disease prevalence, severity, and related research in each area.
    """
    try:
        # Read the CSV data
        df = pd.read_csv("rice_disease_analysis/rice_disease_pubmed_data.csv")
        
        # Define major research locations
        major_locations = {
            "TH-C": {"name": "ภาคกลาง, ประเทศไทย", "lat": 13.7563, "lng": 100.5018},
            "TH-N": {"name": "ภาคเหนือ, ประเทศไทย", "lat": 18.7883, "lng": 98.9853},
            "TH-NE": {"name": "ภาคตะวันออกเฉียงเหนือ, ประเทศไทย", "lat": 14.8799, "lng": 102.0132},
            "TH-S": {"name": "ภาคใต้, ประเทศไทย", "lat": 7.8804, "lng": 98.3923},
            "CN": {"name": "จีน", "lat": 35.8617, "lng": 104.1954},
            "IN": {"name": "อินเดีย", "lat": 20.5937, "lng": 78.9629},
            "JP": {"name": "ญี่ปุ่น", "lat": 36.2048, "lng": 138.2529},
            "PH": {"name": "ฟิลิปปินส์", "lat": 12.8797, "lng": 121.7740},
            "VN": {"name": "เวียดนาม", "lat": 14.0583, "lng": 108.2772},
            "ID": {"name": "อินโดนีเซีย", "lat": -0.7893, "lng": 113.9213}
        }

        research_locations = []
        
        for loc_id, location in major_locations.items():
            # Simulate location-specific data
            # In a real application, this would come from actual data analysis
            
            # Generate common rice diseases for the location
            diseases = [
                "โรคไหม้",
                "โรคขอบใบแห้ง",
                "โรคใบจุดสีน้ำตาล",
                "โรคกาบใบแห้ง"
            ]
            random.shuffle(diseases)
            diseases = diseases[:random.randint(2, 4)]  # Each location gets 2-4 diseases
            
            # Calculate severity based on various factors
            severity = round(random.uniform(3, 9), 1)  # Scale of 0-10
            
            # Count related research papers
            research_count = random.randint(10, 100)
            
            # Generate recommendations based on diseases
            recommendations = get_location_recommendations(diseases)
            
            # Generate sample related research
            related_research = []
            for _ in range(3):  # 3 related papers per location
                year = str(random.randint(2018, 2023))
                related_research.append(RelatedResearch(
                    title=f"การศึกษา{random.choice(diseases)}ในพื้นที่ {location['name']}",
                    authors=[f"นักวิจัย {i+1}" for i in range(random.randint(1, 3))],
                    year=year
                ))
            
            # Create location object
            research_location = ResearchLocation(
                id=loc_id,
                name=location["name"],
                lat=location["lat"],
                lng=location["lng"],
                diseases=diseases,
                severity=severity,
                researchCount=research_count,
                trends=generate_disease_trends(),
                recommendations=recommendations,
                relatedResearch=related_research
            )
            
            research_locations.append(research_location)
        
        return research_locations
        
    except Exception as e:
        logging.error(f"Error in get_research_locations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching research locations: {str(e)}")

# Initialize the translator globally at the module level
translator = None

def get_translator():
    """Get or initialize the translator."""
    global translator
    if translator is None:
        try:
            translator = Translator(service_urls=['translate.google.com'])
            return translator
        except Exception as e:
            logging.error(f"Error initializing translator: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize translator"
            )
    return translator

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    Translate text to the target language using googletrans.
    """
    try:
        if not request.text or not request.text.strip():
            return {
                "translated_text": request.text,
                "source_language": "auto"
            }

        trans = get_translator()
        logging.info(f"Translating text to {request.target_language}: {request.text[:100]}...")
        
        result = trans.translate(
            text=request.text,
            dest=request.target_language
        )
        
        return {
            "translated_text": result.text,
            "source_language": result.src
        }
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        try:
            # Try reinitializing translator
            global translator
            translator = Translator(service_urls=['translate.google.com'])
            result = translator.translate(
                text=request.text,
                dest=request.target_language
            )
            return {
                "translated_text": result.text,
                "source_language": result.src
            }
        except Exception as e:
            logging.error(f"Translation retry failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Translation error: {str(e)}"
            )

@app.post("/translate-bulk")
async def translate_bulk(request: BulkTranslationRequest):
    """
    Translate multiple texts to the target language using googletrans.
    """
    try:
        if not request.texts:
            return {"translations": []}

        # Filter out empty texts
        texts_to_translate = [text for text in request.texts if text and text.strip()]
        if not texts_to_translate:
            return {"translations": []}

        trans = get_translator()
        translations = []
        
        # Translate texts in smaller batches
        batch_size = 5
        for i in range(0, len(texts_to_translate), batch_size):
            batch = texts_to_translate[i:i + batch_size]
            try:
                for text in batch:
                    result = trans.translate(
                        text=text,
                        dest=request.target_language
                    )
                    translations.append({
                        "translated_text": result.text,
                        "source_language": result.src
                    })
            except Exception as e:
                logging.error(f"Error translating batch {i//batch_size + 1}: {str(e)}")
                # Try reinitializing translator for this batch
                translator = Translator(service_urls=['translate.google.com'])
                for text in batch:
                    result = translator.translate(
                        text=text,
                        dest=request.target_language
                    )
                    translations.append({
                        "translated_text": result.text,
                        "source_language": result.src
                    })

        return {"translations": translations}
    except Exception as e:
        logging.error(f"Bulk translation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk translation error: {str(e)}"
        )

def get_human_friendly_cluster_explanations(cluster_terms: List[List[str]]) -> List[dict]:
    """
    Generate human-friendly explanations for each cluster based on their top terms.
    """
    explanations = []
    
    for idx, terms in enumerate(cluster_terms):
        explanation = {
            "cluster_number": idx,
            "title": "",
            "main_focus": "",
            "explanation": "",
            "practical_use": ""
        }
        
        # Analyze terms to determine cluster focus
        terms_str = " ".join(terms)
        
        if any(term in terms_str for term in ["gene", "resistance", "variety", "breeding"]):
            explanation.update({
                "title": "Rice Plant Protection and Breeding",
                "main_focus": "Development of disease-resistant rice varieties through genetic research",
                "explanation": "This group focuses on creating stronger rice varieties that can naturally resist diseases. "
                              "It includes research on protective genes and breeding techniques.",
                "practical_use": "Helps farmers choose better rice varieties that are less likely to get diseases."
            })
        elif any(term in terms_str for term in ["oswrky", "defense", "expression", "pathogen"]):
            explanation.update({
                "title": "Plant Defense Systems",
                "main_focus": "Understanding how rice plants defend against diseases",
                "explanation": "This research looks at how rice plants fight off diseases naturally, "
                              "especially focusing on defense genes and immune responses.",
                "practical_use": "Helps develop better ways to boost rice plants' natural protection systems."
            })
        elif any(term in terms_str for term in ["protein", "blast", "oryzae", "pathogen"]):
            explanation.update({
                "title": "Disease and Rice Interaction",
                "main_focus": "Study of how diseases attack rice plants",
                "explanation": "This research examines how diseases infect rice plants and what happens during infection. "
                              "It helps understand the disease process better.",
                "practical_use": "Helps predict and prevent disease outbreaks more effectively."
            })
        elif any(term in terms_str for term in ["model", "growth", "disease", "development"]):
            explanation.update({
                "title": "Disease Patterns and Growth Impact",
                "main_focus": "How diseases affect rice growth and spread",
                "explanation": "This research tracks disease spread patterns and their effects on rice plant growth. "
                              "It includes prediction models and growth analysis.",
                "practical_use": "Helps farmers understand how diseases might spread and affect their crops."
            })
        elif any(term in terms_str for term in ["leaf", "plant", "expression", "activity"]):
            explanation.update({
                "title": "Plant Response and Treatment",
                "main_focus": "Identifying disease symptoms and treatments",
                "explanation": "This research focuses on how rice plants show signs of disease and possible treatments. "
                              "It includes studying leaf symptoms and plant responses.",
                "practical_use": "Helps farmers identify diseases early and choose the right treatments."
            })
        else:
            explanation.update({
                "title": f"Research Cluster {idx}",
                "main_focus": "General rice disease research",
                "explanation": "This group contains various studies about rice diseases and their management.",
                "practical_use": "Provides general knowledge about rice disease management."
            })
            
        explanations.append(explanation)
    
    return explanations

def get_disease_solutions() -> dict:
    """
    Provides practical solutions and treatments for common rice diseases.
    """
    return {
        "โรคไหม้": {
            "symptoms": "ใบมีจุดสีน้ำตาลคล้าย รูปตา มีขอบสีน้ำตาลเข้ม ตรงกลางสีเทา",
            "solutions": [
                "1. ใช้พันธุ์ข้าวต้านทานโรค เช่น กข6, กข15, สุพรรณบุรี1",
                "2. คลุกเมล็ดพันธุ์ด้วยสารป้องกันกำจัดเชื้อรา",
                "3. แบ่งใส่ปุ๋ยไนโตรเจน 3 ครั้ง",
                "4. กำจัดวัชพืชในนาข้าว",
                "5. ฉีดพ่นสารป้องกันกำจัดเชื้อราเมื่อพบโรคระบาด"
            ],
            "prevention": [
                "1. ไม่ควรปลูกข้าวแน่นเกินไป",
                "2. หลีกเลี่ยงการใส่ปุ๋ยไนโตรเจนมากเกินไป",
                "3. กำจัดข้าวเรื้อและพืชอาศัย",
                "4. ตรวจแปลงนาอย่างสม่ำเสมอ"
            ]
        },
        "โรคขอบใบแห้ง": {
            "symptoms": "ใบมีสีเหลืองและแห้งตายจากขอบใบ มีหยดน้ำเล็กๆ สีขาวขุ่นบริเวณขอบใบ",
            "solutions": [
                "1. ใช้เมล็ดพันธุ์ปลอดโรค",
                "2. ลดการใส่ปุ๋ยไนโตรเจน",
                "3. ระบายน้ำในแปลงนาให้ทั่วถึง",
                "4. พ่นสารเคมีป้องกันกำจัดเชื้อแบคทีเรียตามคำแนะนำ"
            ],
            "prevention": [
                "1. ไม่ใช้ปุ๋ยไนโตรเจนมากเกินไป",
                "2. รักษาระดับน้ำในนาให้เหมาะสม",
                "3. กำจัดวัชพืชและพืชอาศัย",
                "4. เก็บเกี่ยวในระยะพลับพลึง"
            ]
        },
        "โรคใบจุดสีน้ำตาล": {
            "symptoms": "จุดสีน้ำตาลเล็กๆ กระจายทั่วใบ รูปร่างกลมหรือรี",
            "solutions": [
                "1. ใช้พันธุ์ต้านทาน",
                "2. ปรับปรุงดินด้วยการใส่ปูนขาว",
                "3. ใส่ปุ๋ยโพแทสเซียมเพื่อเพิ่มความต้านทาน",
                "4. ฉีดพ่นสารป้องกันกำจัดเชื้อราตามความจำเป็น"
            ],
            "prevention": [
                "1. ไม่ปลูกข้าวแน่นเกินไป",
                "2. รักษาความสมดุลของธาตุอาหาร",
                "3. กำจัดวัชพืชในนาข้าว",
                "4. ตากดินและไถกลบตอซัง"
            ]
        },
        "โรคกาบใบแห้ง": {
            "symptoms": "แผลสีน้ำตาลแดงที่กาบใบ มีขอบแผลสีน้ำตาลเข้ม",
            "solutions": [
                "1. ลดความหนาแน่นของการปลูก",
                "2. ควบคุมระดับน้ำในนาให้เหมาะสม",
                "3. ใส่ปุ๋ยให้สมดุล",
                "4. ฉีดพ่นสารป้องกันกำจัดเชื้อราในระยะกำเนิดช่อดอก"
            ],
            "prevention": [
                "1. ใช้เมล็ดพันธุ์สะอาด",
                "2. ไม่ใส่ปุ๋ยไนโตรเจนมากเกินไป",
                "3. กำจัดวัชพืชในแปลงนา",
                "4. เก็บเกี่ยวในระยะที่เหมาะสม"
            ]
        },
        "โรคเมล็ดด่าง": {
            "symptoms": "เมล็ดมีจุดสีน้ำตาลหรือดำ เมล็ดลีบ คุณภาพต่ำ",
            "solutions": [
                "1. ใช้เมล็ดพันธุ์สะอาด",
                "2. พ่นสารป้องกันกำจัดเชื้อราในระยะออกดอก",
                "3. เก็บเกี่ยวเมื่อข้าวแก่พอดี",
                "4. ทำความสะอาดเมล็ดพันธุ์ก่อนเก็บรักษา"
            ],
            "prevention": [
                "1. หลีกเลี่ยงการปลูกข้าวในช่วงที่มีฝนชุก",
                "2. ควบคุมความชื้นในแปลงนา",
                "3. กำจัดวัชพืชที่เป็นพืชอาศัย",
                "4. ตากเมล็ดให้แห้งดีก่อนเก็บรักษา"
            ]
        }
    }

@app.get("/disease-solutions")
def get_disease_treatment_methods():
    """
    Get practical solutions for different rice diseases.
    """
    return get_disease_solutions()
