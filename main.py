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
from typing import List, Optional

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
    return k_range, sse, silhouette_scores

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
    background_tasks.add_task(process_data, query.query, query.max_results)
    return {"message": "Analysis started. Please check the results after completion."}

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
    k_range, sse, silhouette_scores = determine_optimal_clusters(X)

    # Plot Elbow and Silhouette Scores
    plt.figure(figsize=(14,6))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Distances (SSE)")
    plt.title("Elbow Method For Optimal k")

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', color='orange')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores For Various k")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "clustering_plots.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Clustering plots saved to {plot_path}.")

    # Choose optimal_k (สามารถปรับปรุงให้เลือกอัตโนมัติตามแนวโน้มของกราฟ)
    optimal_k = 5
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
