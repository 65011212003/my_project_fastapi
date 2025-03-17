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

from deep_translator import GoogleTranslator

# Advanced model imports
import pickle
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from spacy.matcher import PhraseMatcher
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict, Counter
import traceback

# Try to import advanced embedding models
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    has_transformers = True
except ImportError:
    has_transformers = False
    
try:
    import umap
    has_umap = True
except ImportError:
    has_umap = False

try:
    from sentence_transformers import SentenceTransformer
    has_sentence_transformers = True
except ImportError:
    has_sentence_transformers = False

# Add this near the top of your file, after imports
from functools import lru_cache

# Global cache for dataframes with timestamp
df_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

def get_cached_dataframe(file_path, force_refresh=False):
    """
    Cached dataframe loader to improve performance.
    
    Args:
        file_path (str): Path to the CSV file
        force_refresh (bool): Whether to force a refresh of the cache
        
    Returns:
        pandas.DataFrame: The loaded dataframe
    """
    current_time = time.time()
    
    # Check if we need to refresh
    if (force_refresh or 
        file_path not in df_cache or 
        current_time - df_cache[file_path]['timestamp'] > CACHE_EXPIRY):
        
        # Load the dataframe
        try:
            df = pd.read_csv(file_path)
            df_cache[file_path] = {
                'data': df,
                'timestamp': current_time
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading data from {file_path}: {str(e)}"
            )
    
    return df_cache[file_path]['data'].copy()

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
# Advanced Model Configuration
# -----------------------------------
ADVANCED_CONFIG = {
    'pubmed_cache_file': "pubmed_cache.pkl",
    'batch_size': 500,
    'max_articles': 2000,
    'retries': 5,
    'timeout': 60,
    'random_state': 42,
}

# Initialize advanced NLP components
disease_terms = [
    "rice blast", "bacterial blight", "rice stripe virus",
    "brown spot", "sheath blight", "bacterial leaf streak",
    "tungro virus", "bakanae disease", "stem rot",
    "false smut", "rice yellow mottle virus"
]

treatment_terms = [
    "fungicide", "pesticide", "integrated pest management",
    "biocontrol", "chemical control", "cultural practices",
    "resistant varieties", "crop rotation", "seed treatment",
    "biological control", "disease-resistant cultivars"
]

symptom_terms = [
    "leaf spotting", "wilting", "stunted growth",
    "chlorosis", "necrosis", "lesions",
    "yellowing", "discoloration", "leaf blight",
    "leaf streak", "rotting", "grain discoloration"
]

effects_terms = [
    "yield loss", "crop damage", "economic loss",
    "reduced grain quality", "harvest failure", "growth inhibition",
    "production decline", "food security threat"
]

# Initialize PhraseMatcher
advanced_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
advanced_matcher.add("DISEASE", [nlp.make_doc(term) for term in disease_terms])
advanced_matcher.add("TREATMENT", [nlp.make_doc(term) for term in treatment_terms])
advanced_matcher.add("SYMPTOM", [nlp.make_doc(term) for term in symptom_terms])
advanced_matcher.add("EFFECT", [nlp.make_doc(term) for term in effects_terms])

# Initialize advanced embedding models if available
tokenizer = None
model = None
sentence_model = None
device = None

if has_transformers:
    try:
        # Load SciBERT model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logging.info(f"SciBERT model loaded successfully on {device}")
    except Exception as e:
        logging.error(f"Error loading SciBERT model: {e}")
        tokenizer = None
        model = None

# Fallback to SentenceTransformer if transformers not available
if model is None and has_sentence_transformers:
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("SentenceTransformer model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer model: {e}")
        sentence_model = None

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

class AdvancedAnalysisRequest(BaseModel):
    query: Optional[str] = None
    max_results: Optional[int] = 2000
    use_cache: bool = True
    embedding_method: str = "auto"  # "scibert", "sentence_transformer", or "auto"
    clustering_method: str = "kmeans"  # "kmeans" or "dbscan"
    dimension_reduction: str = "auto"  # "umap", "lda", or "auto"

class AdvancedClusterInfo(BaseModel):
    cluster_id: int
    size: int
    common_diseases: Optional[Dict[str, int]] = None
    common_treatments: Optional[Dict[str, int]] = None
    common_symptoms: Optional[Dict[str, int]] = None
    common_effects: Optional[Dict[str, int]] = None
    
class AdvancedAnalysisResponse(BaseModel):
    message: str
    status: str
    clusters_count: int
    total_articles: int
    clusters: Optional[List[AdvancedClusterInfo]] = None

class AdvancedArticle(BaseModel):
    PMID: str
    Title: str
    Abstract: str
    Processed_Abstract: str
    Cluster: int
    year: Optional[str] = None
    Entities: dict

class AdvancedPaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[AdvancedArticle]

# Add this to your Pydantic models section
class DiseaseSolution(BaseModel):
    symptoms: str
    solutions: List[str]
    prevention: List[str]

# Add to Pydantic models section
class ClusterSummary(BaseModel):
    cluster: int
    size: int
    common_diseases: Optional[Dict[str, int]] = None
    common_treatments: Optional[Dict[str, int]] = None
    common_symptoms: Optional[Dict[str, int]] = None
    common_effects: Optional[Dict[str, int]] = None

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

def extract_entities(text: str) -> dict:
    """
    Extract entities from text using spaCy patterns and rules.
    """
    patterns = {
        "DISEASE": [
            ["disease", "infection", "virus", "fungus"],
            ["ADJ", "NOUN"]
        ],
        "SYMPTOM": [
            ["symptom", "effect", "spot", "lesion"],
            ["NOUN"]
        ]
    }
    
    doc = nlp(text)
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    
    for label, (keywords, pos) in patterns.items():
        patterns = [{"label": label, "pattern": [{"LOWER": {"IN": keywords}, "POS": {"IN": pos}}]}]
        ruler.add_patterns(patterns)
    
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    
    return dict(entities)

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
    Determine the optimal number of clusters using Silhouette Score.
    """
    best_k = k_min
    best_score = -1
    silhouette_scores = []
    
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue  # Skip if only 1 cluster
            
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_k = k
            
    logging.info(f"Optimal clusters: {best_k} with score {best_score:.2f}")
    return best_k, silhouette_scores

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
# Advanced Model Utility Functions
# -----------------------------------
def build_advanced_search_query():
    """Builds a comprehensive search query for rice diseases."""
    base_terms = [
        "rice diseases", "rice pathogens", "rice disease symptoms",
        "rice disease management", "rice crop protection", "rice disease control",
        "rice disease resistance", "rice disease epidemiology", "rice disease diagnosis",
        "rice disease prevention", "rice disease treatment", "rice disease impact",
        "rice disease detection", "rice crop diseases", "rice plant diseases",
        "rice disease identification", "rice disease assessment", "rice disease monitoring"
    ]
    disease_specific = [f'"{disease}"' for disease in disease_terms]
    treatment_specific = [f'"{treatment} rice disease"' for treatment in treatment_terms[:5]]
    
    query = " OR ".join(
        [f'"{term}"' for term in base_terms] + 
        disease_specific +
        treatment_specific
    )
    return query

def parse_article_with_bs4(article):
    """Extracts title, abstract, PMID, and publication year from a PubMedArticle using BeautifulSoup."""
    abstract_tag = article.find('AbstractText')
    title_tag = article.find('ArticleTitle')
    pmid_tag = article.find('PMID')
    year_tag = article.find('PubDate').find('Year') if article.find('PubDate') else None
    
    return {
        "Title": title_tag.text if title_tag else "No Title",
        "Abstract": abstract_tag.text if abstract_tag else "",
        "PMID": pmid_tag.text if pmid_tag else "",
        "year": year_tag.text if year_tag else ""
    }

def fetch_pubmed_data_advanced():
    """
    Fetches comprehensive PubMed data on rice diseases using advanced methods.
    Includes validation steps and error handling.
    """
    cache_file = ADVANCED_CONFIG['pubmed_cache_file']
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        if not df.empty:
            logging.info(f"Loaded {len(df)} articles from cache")
            return df

    query = build_advanced_search_query()
    
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0, usehistory="y")
    record = Entrez.read(handle)
    total_count = int(record["Count"])
    webenv = record["WebEnv"]
    query_key = record["QueryKey"]

    logging.info(f"Found {total_count} articles matching the search criteria")

    articles = []
    for start in range(0, min(ADVANCED_CONFIG['max_articles'], total_count), ADVANCED_CONFIG['batch_size']):
        for attempt in range(1, ADVANCED_CONFIG['retries'] + 1):
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    retstart=start,
                    retmax=ADVANCED_CONFIG['batch_size'],
                    webenv=webenv,
                    query_key=query_key,
                    retmode="xml",
                    timeout=ADVANCED_CONFIG['timeout']
                )
                article_data = handle.read()
                soup = BeautifulSoup(article_data, "lxml-xml")
                
                with ThreadPoolExecutor(max_workers=8) as executor:
                    batch_results = list(executor.map(parse_article_with_bs4, soup.find_all('PubmedArticle')))
                
                articles.extend(batch_results)
                logging.info(f"Fetched articles {start} to {start + len(batch_results)}")
                break
            except Exception as e:
                logging.error(f"Attempt {attempt} for batch starting at {start} failed: {e}")
                time.sleep(2 ** attempt)
        else:
            logging.error(f"All attempts failed for batch starting at {start}")

    df = pd.DataFrame(articles)
    
    # Validate the data
    if not df.empty:
        df = df[df['Abstract'].str.len() > 100]  # Remove entries with very short abstracts
        df = df.drop_duplicates(subset=['Abstract'])  # Remove duplicates
        
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        logging.info(f"Successfully collected and validated {len(df)} unique articles")
    else:
        logging.error("No valid articles were collected")
        
    return df

def preprocess_text_improved(text):
    """
    Enhanced text preprocessing with domain-specific considerations.
    """
    if not text or not isinstance(text, str):
        return ""
        
    doc = nlp(text.lower())
    # Keep domain-specific terms intact while removing general stopwords
    tokens = [
        token.lemma_ for token in doc 
        if (token.is_alpha and 
            token.text not in stopwords.words('english')) or 
        token.text in disease_terms + treatment_terms + symptom_terms
    ]
    return " ".join(tokens)

def extract_entities_advanced(text):
    """
    Extract entities from text with improved error handling and normalization.
    
    Args:
        text (str): Input text to process
        
    Returns:
        dict: Dictionary of extracted entities by category
    """
    if not text or not isinstance(text, str):
        return {'DISEASE': {}, 'TREATMENT': {}, 'SYMPTOM': {}, 'EFFECT': {}}
    
    try:
        # Normalize text - remove excess whitespace
        text = ' '.join(text.split())
        
        # Initialize entity dictionary
        entities = {
            'DISEASE': {},
            'TREATMENT': {},
            'SYMPTOM': {},
            'EFFECT': {}
        }
        
        # Process with spaCy if available
        if 'nlp' in globals():
            doc = nlp(text)
            
            # Extract entities from spaCy
            for ent in doc.ents:
                # Categorize entities based on label
                if ent.label_ in ['DISEASE', 'CONDITION', 'DISORDER']:
                    entity_type = 'DISEASE'
                elif ent.label_ in ['TREATMENT', 'PROCEDURE', 'MEDICATION']:
                    entity_type = 'TREATMENT'
                elif ent.label_ in ['SYMPTOM', 'SIGN']:
                    entity_type = 'SYMPTOM'
                elif ent.label_ in ['EFFECT', 'SIDE_EFFECT']:
                    entity_type = 'EFFECT'
                else:
                    continue
                
                # Normalize entity text
                entity_text = ent.text.lower().strip()
                if entity_text:
                    if entity_text in entities[entity_type]:
                        entities[entity_type][entity_text] += 1
                    else:
                        entities[entity_type][entity_text] = 1
        
        return entities
    except Exception as e:
        print(f"Error in entity extraction: {str(e)}")
        return {'DISEASE': {}, 'TREATMENT': {}, 'SYMPTOM': {}, 'EFFECT': {}}

def get_scibert_embeddings(texts, batch_size=32):
    """
    Generates embeddings using SciBERT with batching and progress tracking.
    """
    if model is None or tokenizer is None:
        return get_sentence_embeddings(texts)
        
    embeddings = []
    logging.info("Generating SciBERT embeddings...")
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and move to device
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Get model outputs
            outputs = model(**encoded)
            
            # Use CLS token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 100 == 0:
                logging.info(f"Processed {i + batch_size}/{len(texts)} texts")
    
    return np.array(embeddings)

def get_sentence_embeddings(texts):
    """
    Fallback method for sentence embeddings if SciBERT is not available.
    """
    if sentence_model is None:
        logging.warning("Neither SciBERT nor SentenceTransformer is available.")
        return None
    logging.info("Generating sentence embeddings...")
    embeddings = sentence_model.encode(texts, show_progress_bar=True)
    return embeddings

def reduce_dimensions(X, method='lda', n_components=50):
    """
    Reduces dimensionality using specified method (UMAP or LDA).
    """
    if method == 'umap' and has_umap:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.1,
            random_state=ADVANCED_CONFIG['random_state']
        )
        return reducer.fit_transform(X)
    elif method == 'lda':
        lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=ADVANCED_CONFIG['random_state']
        )
        return lda.fit_transform(X)
    else:
        logging.warning(f"Method {method} not available. Falling back to LDA.")
        lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=ADVANCED_CONFIG['random_state']
        )
        return lda.fit_transform(X)

def cluster_data_advanced(X, method='kmeans', n_clusters=None):
    """
    Clusters data using either KMeans or DBSCAN.
    """
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(X)[0]
        
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=ADVANCED_CONFIG['random_state'])
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
    
    labels = model.fit_predict(X)
    return labels, model

def summarize_clusters_advanced(df):
    """
    Generate summaries for each cluster with improved entity extraction.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the article data with clusters
        
    Returns:
        pandas.DataFrame: A DataFrame with cluster summaries
    """
    # Ensure we have a cluster column
    if 'Cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column")
        
    # Group by cluster
    cluster_groups = df.groupby('Cluster')
    
    # Create summary dataframe
    summaries = []
    
    for cluster, group in cluster_groups:
        # Get cluster size
        size = len(group)
        
        # Extract entity counts
        disease_counts = Counter()
        treatment_counts = Counter()
        symptom_counts = Counter()
        effect_counts = Counter()
        
        # Process entities for each article in the cluster
        for _, row in group.iterrows():
            if isinstance(row.get('Entities'), dict):
                entities = row['Entities']
                disease_counts.update(entities.get('DISEASE', {}))
                treatment_counts.update(entities.get('TREATMENT', {}))
                symptom_counts.update(entities.get('SYMPTOM', {}))
                effect_counts.update(entities.get('EFFECT', {}))
                
        # Create summary record
        summary = {
            'cluster': int(cluster),
            'size': size,
            'common_diseases': dict(disease_counts.most_common(10)),
            'common_treatments': dict(treatment_counts.most_common(10)),
            'common_symptoms': dict(symptom_counts.most_common(10)),
            'common_effects': dict(effect_counts.most_common(10))
        }
        
        summaries.append(summary)
        
    return pd.DataFrame(summaries)

# -----------------------------------
# API Endpoints
# -----------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
def analyze_data(query: PubMedQuery, background_tasks: BackgroundTasks):
    """
    Initiate the analysis process with the given PubMed query.
    The processing is done in the background using the advanced analysis pipeline.
    """
    background_tasks.add_task(process_advanced_data, query.query, query.max_results)
    return {"message": "Analysis started. Please check the results after completion."}

def process_advanced_data(query: str, max_results: int):
    """
    Process PubMed data with enhanced error handling and logging.
    
    Args:
        query (str): PubMed search query
        max_results (int): Maximum number of results to retrieve
        
    Returns:
        str: Status message
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs("rice_disease_analysis", exist_ok=True)
        
        # Log processing start
        print(f"Starting advanced analysis for query: {query}")
        
        # Fetch data from PubMed
        print(f"Fetching up to {max_results} articles from PubMed...")
        id_list = search_pubmed(query, max_results)
        
        if not id_list:
            return "No articles found matching the query"
            
        print(f"Found {len(id_list)} articles. Fetching details...")
        articles = fetch_details(id_list)
        
        if not articles:
            return "Failed to fetch article details"
            
        # Process articles and create DataFrame
        df = pd.DataFrame(articles)
        
        # Save raw data
        df.to_csv("rice_disease_analysis/raw_data.csv", index=False)
        print(f"Saved {len(df)} raw articles to CSV")
        
        # Preprocess abstracts with progress tracking
        print("Processing abstracts...")
        df['Processed_Abstract'] = df['Abstract'].apply(preprocess_text_improved)
        
        # Extract entities
        print("Extracting entities...")
        df['Entities'] = df['Abstract'].apply(extract_entities_advanced)
        
        # Get embeddings
        processed_texts = df['Processed_Abstract'].tolist()
        print("Generating embeddings...")
        embeddings = get_sentence_embeddings(processed_texts)
        
        # Reduce dimensions and cluster
        print("Reducing dimensions...")
        reduced_embeddings = reduce_dimensions(embeddings, method='umap', n_components=50)
        
        print("Clustering data...")
        optimal_clusters = determine_optimal_clusters(reduced_embeddings)
        labels = cluster_data_advanced(reduced_embeddings, n_clusters=optimal_clusters)
        
        # Add cluster labels to dataframe
        df['Cluster'] = labels
        
        # Save processed data
        df.to_csv("rice_disease_analysis/processed_data.csv", index=False)
        print("Saved processed data with clusters")
        
        # Generate and save cluster summaries
        print("Generating cluster summaries...")
        summaries = summarize_clusters_advanced(df)
        summaries.to_csv("rice_disease_analysis/cluster_summaries.csv", index=False)
        
        return f"Successfully processed {len(df)} articles into {optimal_clusters} clusters"
        
    except Exception as e:
        error_message = f"Error in advanced processing: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return error_message

@app.get("/status", response_model=AdvancedAnalysisResponse)
def get_status():
    """
    Check the status of the analysis.
    """
    output_dir = "rice_disease_analysis"
    data_path = os.path.join(output_dir, "rice_disease_pubmed_data.csv")
    summary_path = os.path.join(output_dir, "cluster_summaries.csv")
    
    if not os.path.exists(data_path):
        return {
            "message": "Analysis has not been run or is still in progress.",
            "status": "not_started",
            "clusters_count": 0,
            "total_articles": 0
        }
    
    df = pd.read_csv(data_path)
    
    if not os.path.exists(summary_path):
        return {
            "message": "Analysis is partially complete. Cluster summary not available yet.",
            "status": "partial",
            "clusters_count": df['Cluster'].nunique(),
            "total_articles": len(df)
        }
    
    # Load cluster summary
    cluster_summary = pd.read_csv(summary_path)
    
    clusters = []
    for _, row in cluster_summary.iterrows():
        cluster_info = {
            "cluster_id": int(row['cluster']),
            "size": int(row['size'])
        }
        
        # Add entity frequencies if available
        for entity_type in ['common_diseases', 'common_treatments', 'common_symptoms', 'common_effects']:
            if entity_type in row and not pd.isna(row[entity_type]):
                try:
                    # Convert string representation of dict to actual dict
                    if isinstance(row[entity_type], str):
                        cluster_info[entity_type] = eval(row[entity_type])
                    else:
                        cluster_info[entity_type] = row[entity_type]
                except:
                    pass
        
        clusters.append(AdvancedClusterInfo(**cluster_info))
    
    return {
        "message": "Analysis complete.",
        "status": "complete",
        "clusters_count": df['Cluster'].nunique(),
        "total_articles": len(df),
        "clusters": clusters
    }

@app.get("/articles/{cluster_id}", response_model=List[AdvancedArticle])
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
        # Convert string representation of entities to dict
        entities = row['Entities']
        if isinstance(entities, str):
            try:
                entities = eval(entities)
            except:
                entities = {}
                
        # Handle year field if present
        year = row.get('year', None)
        
        articles.append(AdvancedArticle(
            PMID=str(row['PMID']),
            Title=row['Title'],
            Abstract=row['Abstract'],
            Processed_Abstract=row['Processed_Abstract'],
            Cluster=int(row['Cluster']),
            year=year,
            Entities=entities
        ))
    return articles

@app.get("/articles", response_model=AdvancedPaginatedResponse)
async def get_articles(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
    """
    Get paginated articles from the dataset.
    """
    try:
        # Load the CSV data
        df = pd.read_csv("rice_disease_analysis/rice_disease_pubmed_data.csv")
        
        # Calculate pagination
        total_articles = len(df)
        total_pages = (total_articles + page_size - 1) // page_size
        
        # Validate page number
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        # Get the slice of data for the current page
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_articles)
        
        # Get the articles for the current page
        page_articles = df.iloc[start_idx:end_idx].to_dict(orient='records')
        
        return {
            "total": total_articles,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "items": page_articles  # Make sure this returns List[AdvancedArticle]
        }
    except Exception as e:
        logging.error(f"Error retrieving articles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving articles: {str(e)}")

@app.get("/statistics")
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
    treatment_counts = {}
    symptom_counts = {}
    effect_counts = {}
    
    for _, row in df.iterrows():
        entities = row['Entities']
        if isinstance(entities, str):
            try:
                entities = eval(entities)
                
                # Count diseases
                for disease in entities.get('DISEASE', []):
                    disease_counts[disease] = disease_counts.get(disease, 0) + 1
                
                # Count treatments
                for treatment in entities.get('TREATMENT', []):
                    treatment_counts[treatment] = treatment_counts.get(treatment, 0) + 1
                
                # Count symptoms
                for symptom in entities.get('SYMPTOM', []):
                    symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
                
                # Count effects
                for effect in entities.get('EFFECT', []):
                    effect_counts[effect] = effect_counts.get(effect, 0) + 1
                    
            except:
                continue
    
    # Get top 10 for each category
    top_diseases = dict(sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    top_treatments = dict(sorted(treatment_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    top_symptoms = dict(sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    top_effects = dict(sorted(effect_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Extract years from data if available
    yearly_trends = {}
    if 'year' in df.columns:
        year_counts = df['year'].value_counts().to_dict()
        yearly_trends = {str(year): count for year, count in year_counts.items() if str(year).isdigit()}
    else:
        # Try to extract years from abstracts
        pattern = r'\b(19|20)\d{2}\b'
        years = []
        for text in df['Abstract']:
            if isinstance(text, str):
                found_years = re.findall(pattern, text)
                years.extend(found_years)
        
        for year in years:
            yearly_trends[year] = yearly_trends.get(year, 0) + 1
    
    # Get cluster distribution
    cluster_distribution = df['Cluster'].value_counts().to_dict()
    cluster_distribution = {str(cluster): count for cluster, count in cluster_distribution.items()}
    
    return {
        "disease_counts": top_diseases,
        "treatment_counts": top_treatments,
        "symptom_counts": top_symptoms,
        "effect_counts": top_effects,
        "yearly_trends": yearly_trends,
        "cluster_distribution": cluster_distribution,
        "total_articles": len(df),
        "unique_diseases": len(disease_counts),
        "unique_treatments": len(treatment_counts),
        "unique_symptoms": len(symptom_counts),
        "unique_effects": len(effect_counts)
    }

@app.get("/clusters", response_model=List[AdvancedClusterInfo])
def get_clusters():
    """
    Retrieve the cluster information from the analysis.
    
    Returns:
        List[AdvancedClusterInfo]: A list of cluster information objects containing
        cluster ID, size, and common entities (diseases, treatments, symptoms, effects)
    """
    summary_path = "rice_disease_analysis/cluster_summaries.csv"
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Cluster summaries not found. Please run the analysis first.")

    # Use a try-except block for specific file reading errors
    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading cluster data: {str(e)}")
    
    # Convert entity columns from string to dict
    entity_columns = ['common_diseases', 'common_treatments', 'common_symptoms', 'common_effects']
    
    # Process entity columns properly
    for column in entity_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: eval(x) if isinstance(x, str) and x.strip() else {})
    
    # Create a list to store all cluster info
    clusters = []
    
    # Use DataFrame.to_dict with orient='records' for faster conversion
    records = df.to_dict(orient='records')
    
    for record in records:
        cluster_info = AdvancedClusterInfo(
            cluster_id=int(record['cluster']),
            size=int(record['size']),
            common_diseases=record.get('common_diseases', {}),
            common_treatments=record.get('common_treatments', {}),
            common_symptoms=record.get('common_symptoms', {}),
            common_effects=record.get('common_effects', {})
        )
        clusters.append(cluster_info)
    
    return clusters

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
    Get research locations with disease data and recommendations.
    
    Returns:
        List[ResearchLocation]: A list of research locations with associated disease data
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
    try:
        return GoogleTranslator()
    except Exception as e:
        logging.error(f"Error initializing translator: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize translator"
        )

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    Translate text to the target language using deep-translator.
    """
    try:
        if not request.text or not request.text.strip():
            return {
                "translated_text": request.text,
                "source_language": "auto"
            }

        translator = get_translator()
        result = translator.translate(
            text=request.text,
            target=request.target_language
        )
        
        return {
            "translated_text": result,
            "source_language": translator.source
        }
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation error: {str(e)}"
        )

@app.post("/translate-bulk")
async def translate_bulk(request: BulkTranslationRequest):
    """
    Translate multiple texts to the target language using deep-translator.
    """
    try:
        if not request.texts:
            return {"translations": []}

        texts_to_translate = [text for text in request.texts if text and text.strip()]
        if not texts_to_translate:
            return {"translations": []}

        translator = get_translator()
        translations = []
        
        for text in texts_to_translate:
            try:
                result = translator.translate(
                    text=text,
                    target=request.target_language
                )
                translations.append({
                    "translated_text": result,
                    "source_language": translator.source
                })
            except Exception as e:
                logging.error(f"Error translating text: {str(e)}")
                translations.append({
                    "translated_text": text,
                    "source_language": "auto",
                    "error": str(e)
                })

        return {"translations": translations}
    except Exception as e:
        logging.error(f"Bulk translation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk translation error: {str(e)}"
        )

@app.get("/disease-solutions", response_model=Dict[str, DiseaseSolution])
async def get_disease_solutions():
    """
    Get information about disease solutions, symptoms, and prevention methods.
    
    Returns:
        Dict[str, DiseaseSolution]: A dictionary mapping diseases to their solutions
    """
    try:
        solutions = {
            "โรคไหม้ข้าว (Rice Blast)": {
                "symptoms": "แผลจุดสีน้ำตาลคล้ายรูปตาบนใบข้าว ขอบแผลสีน้ำตาลเข้ม ตรงกลางสีเทาหรือขาว ในสภาพแวดล้อมที่เหมาะสมแผลจะขยายและรวมกันทำให้ใบแห้งตาย",
                "solutions": [
                    "ใช้สารเคมีป้องกันกำจัดเชื้อรา เช่น คาร์เบนดาซิม ไตรไซคลาโซล หรือ อิดิเฟนฟอส",
                    "ฉีดพ่นสารชีวภัณฑ์ เช่น เชื้อราไตรโคเดอร์มา",
                    "ควบคุมระดับน้ำในนาให้เหมาะสม ไม่ให้แห้งสลับกับน้ำขัง",
                    "ใส่ปุ๋ยไนโตรเจนในปริมาณที่เหมาะสม ไม่มากเกินไป"
                ],
                "prevention": [
                    "ใช้พันธุ์ข้าวต้านทานโรค",
                    "คลุกเมล็ดพันธุ์ด้วยสารป้องกันกำจัดเชื้อราก่อนปลูก",
                    "กำจัดวัชพืชในนาและบริเวณใกล้เคียง",
                    "ไม่ควรปลูกข้าวหนาแน่นเกินไป",
                    "หลีกเลี่ยงการให้ปุ๋ยไนโตรเจนมากเกินไป"
                ]
            },
            "โรคขอบใบแห้ง (Bacterial Leaf Blight)": {
                "symptoms": "แผลเริ่มจากขอบใบหรือปลายใบ มีลักษณะเป็นทางสีเหลืองและเปลี่ยนเป็นสีน้ำตาล แผลจะลุกลามเข้าไปในใบทำให้ใบแห้งตาย",
                "solutions": [
                    "ไม่มีสารเคมีที่มีประสิทธิภาพในการควบคุมโรคนี้โดยตรง",
                    "ระบายน้ำออกจากแปลงนาเพื่อลดความชื้น",
                    "ใช้ปุ๋ยโพแทสเซียมเพื่อเพิ่มความแข็งแรงให้ต้นข้าว",
                    "ฉีดพ่นสารชีวภัณฑ์ที่มีแบคทีเรียปฏิปักษ์"
                ],
                "prevention": [
                    "ใช้พันธุ์ข้าวต้านทานโรค",
                    "ใช้เมล็ดพันธุ์ที่สะอาดปราศจากเชื้อโรค",
                    "ไม่ใส่ปุ๋ยไนโตรเจนมากเกินไป",
                    "กำจัดวัชพืชและตอซังข้าวที่เป็นแหล่งสะสมเชื้อ",
                    "หลีกเลี่ยงการปลูกข้าวหนาแน่นเกินไป"
                ]
            },
            "โรคใบจุดสีน้ำตาล (Brown Spot)": {
                "symptoms": "แผลจุดสีน้ำตาลรูปกลมหรือรูปไข่กระจายทั่วใบ ขนาดแผลประมาณ 2-5 มิลลิเมตร มีสีน้ำตาลเข้มตรงกลางและสีน้ำตาลอ่อนล้อมรอบ",
                "solutions": [
                    "ใช้สารเคมีป้องกันกำจัดเชื้อรา เช่น แมนโคเซบ คาร์เบนดาซิม หรือ โพรพิโคนาโซล",
                    "ฉีดพ่นสารชีวภัณฑ์ เช่น เชื้อราไตรโคเดอร์มา",
                    "ปรับปรุงความอุดมสมบูรณ์ของดินโดยการใส่ปุ๋ยที่มีธาตุโพแทสเซียมและซิลิกอน",
                    "ระบายน้ำออกจากแปลงนาเพื่อลดความชื้น"
                ],
                "prevention": [
                    "ใช้เมล็ดพันธุ์ที่สะอาดปราศจากโรค",
                    "คลุกเมล็ดพันธุ์ด้วยสารป้องกันกำจัดเชื้อราก่อนปลูก",
                    "ปรับปรุงดินให้มีความอุดมสมบูรณ์",
                    "ใส่ปุ๋ยที่มีธาตุโพแทสเซียมและซิลิกอนเพื่อเพิ่มความแข็งแรงให้ต้นข้าว",
                    "ไถกลบตอซังและฟางข้าวเพื่อลดแหล่งสะสมเชื้อ"
                ]
            },
            "โรคกาบใบแห้ง (Sheath Blight)": {
                "symptoms": "แผลรูปไข่สีเทาอมเขียวหรือสีน้ำตาลบนกาบใบ แผลมีขอบสีน้ำตาลเข้ม แผลจะขยายและรวมกันทำให้กาบใบแห้งตาย",
                "solutions": [
                    "ใช้สารเคมีป้องกันกำจัดเชื้อรา เช่น วาลิดามัยซิน ฟลูโทลานิล หรือ โพรพิโคนาโซล",
                    "ฉีดพ่นสารชีวภัณฑ์ เช่น เชื้อราไตรโคเดอร์มา",
                    "ระบายน้ำออกจากแปลงนาเพื่อลดความชื้น",
                    "ควบคุมวัชพืชในนาข้าวเพื่อลดความชื้นสะสม"
                ],
                "prevention": [
                    "ใช้พันธุ์ข้าวต้านทานโรค",
                    "ไม่ปลูกข้าวหนาแน่นเกินไป",
                    "ไม่ใส่ปุ๋ยไนโตรเจนมากเกินไป",
                    "กำจัดวัชพืชในนาและบริเวณใกล้เคียง",
                    "ไถกลบตอซังและฟางข้าวเพื่อลดแหล่งสะสมเชื้อ"
                ]
            },
            "โรคใบสีส้ม (Tungro Virus)": {
                "symptoms": "ใบข้าวมีสีเหลืองส้ม ต้นข้าวแคระแกร็น ใบสั้นและแคบกว่าปกติ จำนวนรวงและเมล็ดลดลง",
                "solutions": [
                    "ไม่มีสารเคมีที่สามารถกำจัดเชื้อไวรัสได้โดยตรง",
                    "กำจัดแมลงพาหะ (เพลี้ยจักจั่นสีเขียว) โดยใช้สารฆ่าแมลง เช่น อิมิดาคลอพริด หรือ ไทอะมีโทแซม",
                    "ถอนและทำลายต้นข้าวที่เป็นโรคทันที",
                    "ปลูกพืชหมุนเวียนเพื่อตัดวงจรของโรค"
                ],
                "prevention": [
                    "ใช้พันธุ์ข้าวต้านทานโรค",
                    "ปลูกข้าวพร้อมกันในพื้นที่เดียวกันเพื่อตัดวงจรของแมลงพาหะ",
                    "กำจัดวัชพืชที่เป็นแหล่งอาศัยของแมลงพาหะ",
                    "ตรวจแปลงนาอย่างสม่ำเสมอเพื่อพบการระบาดในระยะเริ่มต้น",
                    "ใช้กับดักแมลงเพื่อลดปริมาณแมลงพาหะ"
                ]
            }
        }
        return solutions
    except Exception as e:
        logging.error(f"Error getting disease solutions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting disease solutions: {str(e)}"
        )

@app.get("/cluster-summaries", response_model=List[ClusterSummary])
async def get_cluster_summaries():
    """
    Get summaries of each cluster including common diseases, treatments, symptoms, and effects.
    """
    # Check if cluster summaries file exists
    summary_path = "rice_disease_analysis/cluster_summaries.csv"
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Cluster summaries not found. Please run the analysis first.")

    try:
        df = pd.read_csv(summary_path)
        
        summaries = []
        for _, row in df.iterrows():
            summary = {
                "cluster": int(row['cluster']),
                "size": int(row['size'])
            }
            
            # Add entity frequencies if available
            for entity_type in ['common_diseases', 'common_treatments', 'common_symptoms', 'common_effects']:
                if entity_type in row and not pd.isna(row[entity_type]):
                    try:
                        # Convert string representation of dict to actual dict
                        if isinstance(row[entity_type], str):
                            summary[entity_type] = eval(row[entity_type])
                        else:
                            summary[entity_type] = row[entity_type]
                    except:
                        pass
            
            summaries.append(ClusterSummary(**summary))
        
        return summaries
    except Exception as e:
        logging.error(f"Error loading cluster summaries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading cluster summaries: {str(e)}")

@app.post("/translate", response_model=dict)
async def translate_text(request: TranslationRequest):
    """
    Translate text to the specified language.
    """
    try:
        # Map language codes to Google Translator format
        language_map = {
            "en": "english",
            "th": "thai",
            "zh": "chinese (simplified)"
        }
        
        target_lang = language_map.get(request.target_language, request.target_language)
        
        # Skip translation if text is empty
        if not request.text or not request.text.strip():
            return {"translated_text": request.text}
            
        # Translate the text
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_text = translator.translate(request.text)
        
        return {"translated_text": translated_text}
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.post("/translate-bulk", response_model=dict)
async def translate_bulk(request: BulkTranslationRequest):
    """
    Translate multiple texts to the specified language.
    """
    try:
        # Map language codes to Google Translator format
        language_map = {
            "en": "english",
            "th": "thai",
            "zh": "chinese (simplified)"
        }
        
        target_lang = language_map.get(request.target_language, request.target_language)
        translator = GoogleTranslator(source='auto', target=target_lang)
        
        translations = []
        for text in request.texts:
            # Skip translation if text is empty
            if not text or not isinstance(text, str) or not text.strip():
                translations.append({"original_text": text, "translated_text": text})
                continue
                
            try:
                translated_text = translator.translate(text)
                translations.append({"original_text": text, "translated_text": translated_text})
            except Exception as e:
                logging.error(f"Error translating text: {str(e)}")
                translations.append({"original_text": text, "translated_text": text, "error": str(e)})
        
        return {"translations": translations}
    except Exception as e:
        logging.error(f"Bulk translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk translation error: {str(e)}")



@app.post("/analyze-rice-diseases", response_model=dict)
async def analyze_rice_diseases():
    """
    Analyze rice diseases data using DeepSeek API and return the summary.
    """
    try:
        from openai import OpenAI
        import pandas as pd
        import json
        import ast
        
        client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key="sk-27b612b4c90943bb8566e2f57fd8204d"
        )

        # Load cluster summary data directly
        cluster_summary = pd.read_csv('rice_disease_analysis/cluster_summaries.csv')

        # Initialize lists to store summary data
        diseases_and_impacts = []
        treatments_and_rates = []
        symptoms = []

        # Prepare the message content and collect summary data
        message_content = ""
        for _, row in cluster_summary.iterrows():
            message_content += f"Cluster {row['cluster']} ({row['size']} articles)\n"
            
            # Parse string dictionaries into actual dictionaries, handling empty/null values
            diseases_dict = ast.literal_eval(row['common_diseases']) if pd.notna(row['common_diseases']) else {}
            treatments_dict = ast.literal_eval(row['common_treatments']) if pd.notna(row['common_treatments']) else {}
            symptoms_dict = ast.literal_eval(row['common_symptoms']) if pd.notna(row['common_symptoms']) else {}
            effects_dict = ast.literal_eval(row['common_effects']) if pd.notna(row['common_effects']) else {}
            
            # Collect diseases and impacts
            diseases_and_impacts.extend([(disease, count) for disease, count in diseases_dict.items()])
            message_content += f"Rice Diseases and Impacts: {diseases_dict}\n"
            
            # Collect treatments and success rates
            treatments_and_rates.extend([(treatment, count) for treatment, count in treatments_dict.items()])
            message_content += f"Effective Treatments and Success Rates: {treatments_dict}\n"
            
            # Collect symptoms
            symptoms.extend([(symptom, count) for symptom, count in symptoms_dict.items()])
            message_content += f"Common Symptoms: {symptoms_dict}\n"
            
            # Collect effects
            message_content += f"Observed Effects: {effects_dict}\n\n"

        # Enhanced prompt for rice diseases analysis
        enhanced_prompt = (
            "You are an elite expert in rice disease research and agricultural analytics. "
            "The provided data comprises a detailed clustering analysis of rice disease-related articles, "
            "highlighting the identification of the most common rice diseases along with their impacts, "
            "the compilation of effective treatment methods with associated success rates, "
            "and the discovery of new insights and trends in rice disease research. "
            "Furthermore, please outline a structure for developing a comprehensive database that catalogs rice diseases, their effects, and treatments. "
            "Based on this comprehensive dataset, deliver an in-depth, insightful summary with actionable recommendations for future research and practical applications in rice disease management. "
            "Please ensure that the summary is detailed and covers all aspects of the data provided, including the most common rice diseases, their impacts, effective treatments, common symptoms, observed effects, and emerging trends. "
            "Additionally, provide a thorough analysis of the actionable recommendations, emphasizing the importance of prioritizing research on high-impact diseases, promoting sustainable disease management practices, developing early detection tools, enhancing farmer education and outreach, and building a comprehensive rice disease database. "
            "Your response should be comprehensive and provide a clear roadmap for future research and practical applications in rice disease management.\n\n"
            + message_content
        )

        # Call the DeepSeek API to summarize the output
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an elite expert in rice disease research and agricultural analytics"},
                {"role": "user", "content": enhanced_prompt},
            ],
            stream=False
        )

        # Get the response content
        response_content = response.choices[0].message.content
        summary_data = {
            "Summary": response_content,
            "Collected_Data": {
                "diseases_and_impacts": sorted(diseases_and_impacts, key=lambda x: x[1], reverse=True),
                "treatments_and_rates": sorted(treatments_and_rates, key=lambda x: x[1], reverse=True),
                "symptoms": sorted(symptoms, key=lambda x: x[1], reverse=True)
            }
        }
        
        return summary_data
    except Exception as e:
        logging.error(f"Rice disease analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rice disease analysis error: {str(e)}")

@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring system status.
    
    Returns:
        dict: System health status information
    """
    # Check if key data files exist
    data_files = {
        "raw_data": os.path.exists("rice_disease_analysis/raw_data.csv"),
        "processed_data": os.path.exists("rice_disease_analysis/processed_data.csv"),
        "cluster_summaries": os.path.exists("rice_disease_analysis/cluster_summaries.csv")
    }
    
    # Get Python, library versions
    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "fastapi": fastapi.__version__
    }
    
    # Return status
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_files": data_files,
        "versions": versions,
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

