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
from collections import defaultdict

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
    "rice blast", "bacterial blight", "rice stripe virus", "brown spot", "sheath blight", 
    "bacterial leaf streak", "tungro virus", "bakanae disease", "stem rot", "false smut", 
    "rice yellow mottle virus", "narrow brown leaf spot", "leaf scald", "grain rot", 
    "seedling blight", "downy mildew", "root rot", "leaf smut", "kernel smut", "crown rot", 
    "foot rot", "seedling rot", "sheath rot", "collar rot", "neck blast", "node blast", 
    "panicle blast", "bacterial leaf blight", "bacterial grain rot", "bacterial brown stripe", 
    "bacterial panicle blight", "bacterial sheath brown rot", "red stripe", "black kernel", 
    "stackburn disease", "pecky rice", "straighthead", "bacterial wilt", "bacterial foot rot", 
    "bacterial seedling rot", "rice dwarf virus", "rice black-streaked dwarf virus", 
    "rice ragged stunt virus", "rice grassy stunt virus", "rice transitory yellowing virus", 
    "rice necrosis mosaic virus", "rice gall dwarf virus", "rice hoja blanca virus", 
    "rice orange leaf disease", "rice yellow dwarf disease", "rice yellow stunt disease", 
    "rice yellow mottle virus", "rice stripe necrosis virus", "rice tungro bacilliform virus", 
    "rice tungro spherical virus", "rice yellow mosaic virus", "rice leaf yellowing virus", 
    "rice leaf curl disease", "rice leaf gall virus", "rice leaf virus", "rice leaf mottle virus", 
    "rice leaf streak virus", "rice leaf yellowing virus", "rice necrotic streak virus", 
    "rice necrotic mosaic virus", "rice stripe virus disease", "rice black-streaked dwarf virus disease", 
    "rice dwarf virus disease", "rice gall dwarf virus disease", "rice grassy stunt virus disease", 
    "rice ragged stunt virus disease", "rice stripe virus disease", "rice transitory yellowing virus disease", 
    "rice tungro disease", "rice yellow dwarf disease", "rice yellow mottle virus disease", 
    "rice yellow stunt disease", "rice yellow stripe virus disease", "rice waika virus", 
    "rice water weevil", "rice stem borer", "rice leaf folder", "rice gall midge", 
    "rice hispa", "rice thrips", "rice bug", "rice leaffolder", "rice caseworm", "rice whorl maggot", 
    "rice mealybug", "rice root aphid", "rice root nematode", "rice white tip nematode", "rice root knot nematode"
]

treatment_terms = [
    "fungicide", "pesticide", "integrated pest management", "biocontrol", "chemical control", 
    "cultural practices", "resistant varieties", "crop rotation", "seed treatment", "biological control", 
    "disease-resistant cultivars", "organic farming", "biopesticides", "botanical pesticides", 
    "microbial pesticides", "natural enemies", "predators", "parasitoids", "antagonistic microorganisms", 
    "beneficial fungi", "beneficial bacteria", "beneficial nematodes", "plant extracts", "essential oils", 
    "neem oil", "garlic extract", "chili extract", "turmeric extract", "aloe vera extract", 
    "eucalyptus oil", "clove oil", "cinnamon oil", "lemongrass oil", "citronella oil", 
    "peppermint oil", "thyme oil", "rosemary oil", "lavender oil", "tea tree oil", 
    "systemic acquired resistance", "induced systemic resistance", "plant defense activators", 
    "silicon application", "potassium application", "calcium application", "balanced fertilization", 
    "organic amendments", "compost application", "vermicompost application", "green manuring", 
    "biofertilizers", "mycorrhizal fungi", "rhizobacteria", "azolla", "blue-green algae", 
    "water management", "field sanitation", "weed management", "tillage practices", "mulching", 
    "intercropping", "trap crops", "companion planting", "cover crops", "allelopathic plants", 
    "solarization", "hot water treatment", "thermotherapy", "cryotherapy", "electromagnetic treatment", 
    "ultraviolet radiation", "gamma radiation", "ultrasound treatment", "plasma treatment", 
    "ozone treatment", "electrolyzed water", "acidified water", "alkaline water", "hydrogen peroxide", 
    "chlorine dioxide", "peracetic acid", "quaternary ammonium compounds", "copper compounds", 
    "sulfur compounds", "bicarbonates", "phosphites", "silicates", "plant growth regulators", 
    "plant hormones", "salicylic acid", "jasmonic acid", "ethylene", "brassinosteroids", "strigolactones"
]

symptom_terms = [
    "leaf spotting", "wilting", "stunted growth", "chlorosis", "necrosis", "lesions", 
    "yellowing", "discoloration", "leaf blight", "leaf streak", "rotting", "grain discoloration", 
    "leaf curling", "leaf rolling", "leaf twisting", "leaf crinkling", "leaf puckering", 
    "leaf distortion", "leaf malformation", "leaf mottling", "leaf mosaic", "leaf vein clearing", 
    "leaf vein banding", "leaf vein necrosis", "leaf vein yellowing", "leaf vein reddening", 
    "leaf vein browning", "leaf vein blackening", "leaf margin necrosis", "leaf margin yellowing", 
    "leaf margin reddening", "leaf margin browning", "leaf margin blackening", "leaf tip necrosis", 
    "leaf tip yellowing", "leaf tip reddening", "leaf tip browning", "leaf tip blackening", 
    "leaf spot", "leaf blotch", "leaf fleck", "leaf rust", "leaf smut", "leaf mold", 
    "leaf mildew", "leaf scorch", "leaf scald", "leaf burn", "leaf firing", "leaf bronzing", 
    "leaf purpling", "leaf reddening", "leaf browning", "leaf blackening", "leaf senescence", 
    "leaf abscission", "leaf drop", "leaf shedding", "leaf wilting", "leaf drying", 
    "leaf shriveling", "leaf withering", "leaf dieback", "stem lesions", "stem canker", 
    "stem rot", "stem blight", "stem wilt", "stem dieback", "stem discoloration", 
    "stem necrosis", "stem galls", "stem tumors", "stem swelling", "stem distortion", 
    "stem malformation", "stem cracking", "stem splitting", "stem breakage", "stem lodging", 
    "root rot", "root necrosis", "root discoloration", "root lesions", "root galls", 
    "root knots", "root swelling", "root distortion", "root malformation", "root stunting", 
    "panicle blight", "panicle blast", "panicle rot", "panicle discoloration", "empty panicles"
]

effects_terms = [
    "yield loss", "crop damage", "economic loss", "reduced grain quality", "harvest failure", 
    "growth inhibition", "production decline", "food security threat", "reduced market value", 
    "increased production costs", "reduced export potential", "reduced income", "reduced profitability", 
    "financial hardship", "bankruptcy", "abandoned farmland", "reduced farm viability", 
    "reduced rural livelihoods", "rural poverty", "migration from rural areas", "food shortage", 
    "food price increase", "food insecurity", "malnutrition", "hunger", "starvation", 
    "reduced dietary diversity", "reduced nutritional quality", "reduced caloric intake", 
    "reduced protein intake", "reduced micronutrient intake", "health impacts", "increased vulnerability", 
    "reduced resilience", "increased susceptibility to other stresses", "reduced ability to cope", 
    "reduced adaptive capacity", "reduced recovery potential", "ecosystem impacts", "biodiversity loss", 
    "reduced ecosystem services", "altered species composition", "altered community structure", 
    "altered ecosystem function", "altered nutrient cycling", "altered water cycling", 
    "altered carbon cycling", "altered energy flow", "altered trophic interactions", 
    "altered predator-prey relationships", "altered competition", "altered mutualism", 
    "altered parasitism", "altered herbivory", "altered pollination", "altered seed dispersal", 
    "altered decomposition", "altered soil formation", "altered soil fertility", "altered soil structure", 
    "altered soil biology", "altered soil chemistry", "altered soil physics", "altered soil hydrology", 
    "altered soil temperature", "altered soil moisture", "altered soil aeration", "altered soil pH", 
    "altered soil organic matter", "altered soil carbon", "altered soil nitrogen", "altered soil phosphorus", 
    "altered soil potassium", "altered soil microbiome", "altered soil fauna", "altered soil flora", 
    "altered soil food web", "altered soil health", "altered soil quality", "altered soil productivity", 
    "altered soil sustainability", "altered soil resilience", "altered soil degradation", 
    "altered soil erosion", "altered soil compaction", "altered soil salinization", "altered soil acidification", 
    "altered soil contamination", "altered soil pollution", "altered soil remediation", "altered soil restoration"
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
        "rice disease identification", "rice disease assessment", "rice disease monitoring",
        "rice fungal diseases", "rice bacterial diseases", "rice viral diseases",
        "rice disease outbreak", "rice disease severity", "rice disease distribution",
        "rice disease economic impact", "rice disease yield loss", "rice disease forecasting",
        "rice disease integrated management", "rice disease biocontrol", "rice disease chemical control",
        "rice disease cultural control", "rice disease host resistance", "rice disease molecular detection"
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
    Optimized text preprocessing using parallel processing for larger datasets
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Add multiprocessing for large batches of text
    def process_single_text(text):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove stopwords and lemmatize
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
        return " ".join(tokens)
    
    # For single text processing
    return process_single_text(text)

# Add parallel processing function for batch processing
def preprocess_texts_parallel(texts, n_jobs=-1):
    """
    Process multiple texts in parallel
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    if n_jobs == -1:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        processed_texts = list(executor.map(preprocess_text_improved, texts))
    
    return processed_texts

# Import necessary libraries for caching
from functools import lru_cache

# Add caching to entity extraction function
@lru_cache(maxsize=10000)
def extract_entities_advanced(text):
    """
    Extract entities from text with caching for improved performance.
    """
    if not text or not isinstance(text, str):
        return {"DISEASE": [], "TREATMENT": [], "SYMPTOM": [], "EFFECT": []}
    
    # Convert text to a hashable object for caching
    if isinstance(text, str):
        text_for_processing = text
    else:
        return {"DISEASE": [], "TREATMENT": [], "SYMPTOM": [], "EFFECT": []}
        
    doc = nlp(text_for_processing)
    matches = advanced_matcher(doc)
    extracted = {
        "DISEASE": set(), 
        "TREATMENT": set(), 
        "SYMPTOM": set(),
        "EFFECT": set()
    }
    
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        extracted[label].add(span.text)
        
    return {k: list(v) for k, v in extracted.items()}

def get_scibert_embeddings(texts, batch_size=32):
    """
    Generates embeddings using SciBERT with batching and progress tracking.
    """
    if model is None or tokenizer is None:
        return get_sentence_embeddings(texts)
        
    embeddings = []
    logging.info("Generating SciBERT embeddings...")
    
    # Add device selection with CUDA support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Add progress tracking with tqdm
    from tqdm import tqdm
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and move to device
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move tensors to the selected device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Get model outputs
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Use CLS token embedding (first token) as the sentence embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
            
            # Force garbage collection to free up memory
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
    Optimized dimension reduction with memory efficiency and progress tracking
    """
    logging.info(f"Reducing dimensions using {method} method...")
    
    if method == 'umap':
        import umap
        # Use low-memory mode for UMAP and add random state for reproducibility
        reducer = umap.UMAP(n_components=min(n_components, 100), 
                           low_memory=True, 
                           random_state=42)
        from tqdm import tqdm
        # Show progress during fitting
        with tqdm(total=100, desc="UMAP Dimension Reduction") as pbar:
            class UMAPCallback:
                def __init__(self, pbar):
                    self.pbar = pbar
                    self.previous_epoch = 0
                def __call__(self, reducer, epoch):
                    self.pbar.update(epoch - self.previous_epoch)
                    self.previous_epoch = epoch
            
            callback = UMAPCallback(pbar)
            return reducer.fit_transform(X, callbacks=[callback])
    
    elif method == 'lda':
        # If this is document-term matrix for LDA
        from sklearn.decomposition import LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_components=n_components, 
                                      random_state=42, 
                                      n_jobs=-1,  # Use all available cores
                                      learning_method='online',  # Faster for large datasets
                                      batch_size=128)  # Batch processing 
        return lda.fit_transform(X)
    
    else:  # default to PCA as fallback
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, X.shape[1]))
        return pca.fit_transform(X)

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
    Provides comprehensive cluster summaries with entity frequencies.
    """
    results = []
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster_id]
        texts = cluster_df['Abstract'].tolist()
        entities_list = [extract_entities_advanced(text) for text in texts]

        # Extract all entity types
        entities = {
            entity_type: [e for ent in entities_list for e in ent.get(entity_type, [])]
            for entity_type in ['DISEASE', 'TREATMENT', 'SYMPTOM', 'EFFECT']
        }

        # Calculate frequencies for each entity type
        frequencies = {
            f"common_{k.lower()}s": pd.Series(v).value_counts().head(5).to_dict()
            for k, v in entities.items() if v
        }

        results.append({
            "cluster": cluster_id,
            "size": len(cluster_df),
            **frequencies
        })
        
    return pd.DataFrame(results)

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
    Process data using the advanced analysis pipeline.
    """
    # Import the global model variables
    global model, tokenizer, sentence_model, has_transformers, has_sentence_transformers
    
    logging.info("Starting advanced data processing...")

    # Define output directory
    output_dir = "rice_disease_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data Collection using advanced methods
    df = fetch_pubmed_data_advanced()
    logging.info(f"Collected {len(df)} articles with abstracts.")

    if df.empty:
        logging.error("No articles fetched. Exiting process.")
        return

    initial_count = len(df)

    # 2. Data Preprocessing with improved methods
    df['Processed_Abstract'] = df['Abstract'].apply(preprocess_text_improved)
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

    # 3. Advanced Vectorization and Embedding
    # Choose embedding method based on availability
    if has_transformers and model is not None:
        logging.info("Using SciBERT embeddings")
        embeddings = get_scibert_embeddings(df['Processed_Abstract'].tolist())
    elif has_sentence_transformers and sentence_model is not None:
        logging.info("Using SentenceTransformer embeddings")
        embeddings = get_sentence_embeddings(df['Processed_Abstract'].tolist())
    else:
        logging.info("Using TF-IDF vectorization")
        X, vectorizer, feature_names = vectorize_text(df['Processed_Abstract'])
        embeddings = X

    # 4. Dimensionality Reduction
    if has_umap:
        logging.info("Reducing dimensions with UMAP")
        reduced_embeddings = reduce_dimensions(embeddings, method='umap', n_components=50)
    else:
        logging.info("Reducing dimensions with LDA")
        reduced_embeddings = reduce_dimensions(embeddings, method='lda', n_components=50)

    # 5. Advanced Clustering
    logging.info("Performing advanced clustering")
    clusters, model = cluster_data_advanced(reduced_embeddings)
    df['Cluster'] = clusters

    # 6. Enhanced Entity Extraction
    logging.info("Extracting entities with advanced methods")
    df['Entities'] = df['Abstract'].apply(extract_entities_advanced)

    # 7. Save Results
    data_csv_path = os.path.join(output_dir, "rice_disease_pubmed_data.csv")
    df.to_csv(data_csv_path, index=False)
    logging.info(f"Data saved to {data_csv_path}.")

    # 8. Generate Cluster Summaries
    cluster_summaries = summarize_clusters_advanced(df)
    cluster_summaries_path = os.path.join(output_dir, "cluster_summaries.csv")
    cluster_summaries.to_csv(cluster_summaries_path, index=False)
    logging.info(f"Cluster summaries saved to {cluster_summaries_path}.")

    # 9. Generate Visualization Data
    # Create silhouette score visualization if using k-means
    if hasattr(model, 'n_clusters'):
        optimal_k, silhouette_scores = determine_optimal_clusters(reduced_embeddings)
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Scores For Various k")
        plot_path = os.path.join(output_dir, "clustering_plots.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Clustering plots saved to {plot_path}.")

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
@app.get("/articles", response_model=dict)
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
            "articles": page_articles,
            "pagination": {
                "total": total_articles,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
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
    Retrieve information about each cluster, including the number of articles and entity information.
    """
    summary_path = "rice_disease_analysis/cluster_summaries.csv"
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Cluster summaries not found. Please run the analysis first.")

    df = pd.read_csv(summary_path)
    
    clusters = []
    for _, row in df.iterrows():
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
def get_disease_solutions():
    """
    Get information about rice diseases and their solutions.
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
def get_cluster_summaries():
    """
    Retrieve advanced cluster summaries with entity frequencies
    """
    summary_path = "rice_disease_analysis/cluster_summaries.csv"
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Cluster summaries not found. Please run the analysis first.")

    df = pd.read_csv(summary_path)
    
    summaries = []
    for _, row in df.iterrows():
        # Convert string representations of dictionaries to actual dicts
        summary_data = {"cluster": row['cluster'], "size": row['size']}
        
        for field in ['common_diseases', 'common_treatments', 'common_symptoms', 'common_effects']:
            if pd.notna(row[field]) and row[field].strip():
                try:
                    summary_data[field] = eval(row[field])  # Safe for controlled data
                except:
                    summary_data[field] = {}
            else:
                summary_data[field] = {}
                
        summaries.append(ClusterSummary(**summary_data))
    
    return summaries

NEWS_API_KEY = "fb075743fae14f89a5cc6bc75ff19009"
NEWS_API_URL = "https://newsapi.org/v2/everything"

@app.get("/news", response_model=List[NewsArticle])
async def get_rice_disease_news():
    try:
        # Define the API endpoint and parameters
        url = "https://newsapi.org/v2/everything"
        
        # Check if API key exists
        if not NEWS_API_KEY:
            # Return mock data if no API key is available
            return [
                NewsArticle(
                    title="การจัดการโรคข้าว: แนวทางใหม่สำหรับเกษตรกรไทย",
                    description="งานวิจัยล่าสุดเกี่ยวกับการจัดการโรคข้าวในประเทศไทย",
                    url="https://example.com/rice-disease-management",
                    urlToImage="https://example.com/images/rice-field.jpg",
                    publishedAt="2023-08-15T07:30:00Z",
                    source={"name": "เกษตรไทย"}
                ),
                NewsArticle(
                    title="นวัตกรรมใหม่ในการป้องกันโรคข้าวไหม้",
                    description="เทคโนโลยีล่าสุดช่วยลดการระบาดของโรคข้าวไหม้ได้อย่างมีประสิทธิภาพ",
                    url="https://example.com/rice-blast-prevention",
                    urlToImage="https://example.com/images/rice-blast.jpg",
                    publishedAt="2023-08-10T09:15:00Z",
                    source={"name": "วิทยาศาสตร์เกษตร"}
                ),
                NewsArticle(
                    title="ผลกระทบของการเปลี่ยนแปลงสภาพภูมิอากาศต่อโรคข้าว",
                    description="การศึกษาพบว่าการเปลี่ยนแปลงสภาพภูมิอากาศส่งผลต่อการระบาดของโรคข้าว",
                    url="https://example.com/climate-change-rice-disease",
                    urlToImage="https://example.com/images/climate-rice.jpg",
                    publishedAt="2023-08-05T14:20:00Z",
                    source={"name": "สิ่งแวดล้อมไทย"}
                )
            ]
        
        # If API key exists, fetch real data
        params = {
            "apiKey": NEWS_API_KEY,
            "q": "rice disease OR rice pest OR rice pathogen",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="News API error")
                
                data = await response.json()
                
                if data["status"] != "ok":
                    raise HTTPException(status_code=500, detail=data.get("message", "Unknown error"))
                
                articles = []
                for article in data["articles"]:
                    try:
                        articles.append(NewsArticle(
                            title=article["title"],
                            description=article.get("description", ""),
                            url=article["url"],
                            urlToImage=article.get("urlToImage", ""),
                            publishedAt=article["publishedAt"],
                            source=article["source"]
                        ))
                    except Exception as e:
                        # Skip invalid articles
                        continue
                
                return articles
    except Exception as e:
        # Log the error
        print(f"Error fetching news: {str(e)}")
        # Return mock data in case of any error
        return [
            NewsArticle(
                title="การจัดการโรคข้าว: แนวทางใหม่สำหรับเกษตรกรไทย",
                description="งานวิจัยล่าสุดเกี่ยวกับการจัดการโรคข้าวในประเทศไทย",
                url="https://example.com/rice-disease-management",
                urlToImage="https://example.com/images/rice-field.jpg",
                publishedAt="2023-08-15T07:30:00Z",
                source={"name": "เกษตรไทย"}
            ),
            NewsArticle(
                title="นวัตกรรมใหม่ในการป้องกันโรคข้าวไหม้",
                description="เทคโนโลยีล่าสุดช่วยลดการระบาดของโรคข้าวไหม้ได้อย่างมีประสิทธิภาพ",
                url="https://example.com/rice-blast-prevention",
                urlToImage="https://example.com/images/rice-blast.jpg",
                publishedAt="2023-08-10T09:15:00Z",
                source={"name": "วิทยาศาสตร์เกษตร"}
            ),
            NewsArticle(
                title="ผลกระทบของการเปลี่ยนแปลงสภาพภูมิอากาศต่อโรคข้าว",
                description="การศึกษาพบว่าการเปลี่ยนแปลงสภาพภูมิอากาศส่งผลต่อการระบาดของโรคข้าว",
                url="https://example.com/climate-change-rice-disease",
                urlToImage="https://example.com/images/climate-rice.jpg",
                publishedAt="2023-08-05T14:20:00Z",
                source={"name": "สิ่งแวดล้อมไทย"}
            )
        ]

@app.get("/research-locations", response_model=List[ResearchLocation])
async def get_research_locations():
    """
    Get research locations with disease information.
    This is a simulated endpoint that returns mock data for the map visualization.
    """
    # Simulated research locations data
    locations = [
        {
            "id": "loc1",
            "name": "Bangkok, Thailand",
            "lat": 13.7563, 
            "lng": 100.5018,
            "diseases": ["Rice blast", "Bacterial blight"],
            "severity": 7.5,
            "researchCount": 42,
            "trends": [
                {"year": "2020", "cases": 12},
                {"year": "2021", "cases": 15},
                {"year": "2022", "cases": 10},
                {"year": "2023", "cases": 5}
            ],
            "recommendations": [
                "Use resistant rice varieties",
                "Apply fungicides preventatively",
                "Maintain proper water management"
            ],
            "relatedResearch": [
                {"title": "Genetic diversity of rice blast in Central Thailand", "authors": ["Somchai P.", "Wichit S."], "year": "2022"},
                {"title": "Efficacy of new fungicides against rice blast", "authors": ["Pranee K.", "Tawee L."], "year": "2021"}
            ]
        },
        {
            "id": "loc2",
            "name": "Manila, Philippines",
            "lat": 14.5995, 
            "lng": 120.9842,
            "diseases": ["Bacterial leaf blight", "Tungro virus"],
            "severity": 8.2,
            "researchCount": 38,
            "trends": [
                {"year": "2020", "cases": 18},
                {"year": "2021", "cases": 22},
                {"year": "2022", "cases": 15},
                {"year": "2023", "cases": 10}
            ],
            "recommendations": [
                "Plant resistant varieties",
                "Control insect vectors",
                "Practice crop rotation"
            ],
            "relatedResearch": [
                {"title": "Tungro virus resistance in Philippine rice varieties", "authors": ["Maria C.", "Juan D."], "year": "2022"},
                {"title": "Vector dynamics of rice tungro disease", "authors": ["Elena F.", "Roberto G."], "year": "2021"}
            ]
        },
        {
            "id": "loc3",
            "name": "New Delhi, India",
            "lat": 28.6139, 
            "lng": 77.2090,
            "diseases": ["Brown spot", "Sheath blight"],
            "severity": 6.8,
            "researchCount": 56,
            "trends": [
                {"year": "2020", "cases": 25},
                {"year": "2021", "cases": 20},
                {"year": "2022", "cases": 18},
                {"year": "2023", "cases": 15}
            ],
            "recommendations": [
                "Use balanced fertilization",
                "Apply fungicides at early disease onset",
                "Improve drainage in fields"
            ],
            "relatedResearch": [
                {"title": "Climate change impact on brown spot disease in North India", "authors": ["Rajesh K.", "Priya S."], "year": "2022"},
                {"title": "Integrated management of sheath blight in rice", "authors": ["Amit P.", "Sunita R."], "year": "2021"}
            ]
        },
        {
            "id": "loc4",
            "name": "Hanoi, Vietnam",
            "lat": 21.0285, 
            "lng": 105.8542,
            "diseases": ["Rice blast", "Bacterial leaf streak"],
            "severity": 7.2,
            "researchCount": 35,
            "trends": [
                {"year": "2020", "cases": 15},
                {"year": "2021", "cases": 18},
                {"year": "2022", "cases": 12},
                {"year": "2023", "cases": 8}
            ],
            "recommendations": [
                "Use silicon fertilizers",
                "Apply copper-based bactericides",
                "Implement proper field sanitation"
            ],
            "relatedResearch": [
                {"title": "Silicon application for rice blast management in Vietnam", "authors": ["Nguyen V.", "Tran H."], "year": "2022"},
                {"title": "Bacterial leaf streak epidemiology in Red River Delta", "authors": ["Le T.", "Pham D."], "year": "2021"}
            ]
        },
        {
            "id": "loc5",
            "name": "Yangon, Myanmar",
            "lat": 16.8661, 
            "lng": 96.1951,
            "diseases": ["False smut", "Stem rot"],
            "severity": 5.9,
            "researchCount": 28,
            "trends": [
                {"year": "2020", "cases": 10},
                {"year": "2021", "cases": 12},
                {"year": "2022", "cases": 15},
                {"year": "2023", "cases": 18}
            ],
            "recommendations": [
                "Apply fungicides at heading stage",
                "Reduce nitrogen application",
                "Improve water management"
            ],
            "relatedResearch": [
                {"title": "False smut emergence in Myanmar rice fields", "authors": ["Aung S.", "Khin M."], "year": "2022"},
                {"title": "Management strategies for stem rot in rice", "authors": ["Tun L.", "Win N."], "year": "2021"}
            ]
        },
        {
            "id": "loc6",
            "name": "Jakarta, Indonesia",
            "lat": -6.2088, 
            "lng": 106.8456,
            "diseases": ["Rice yellow mottle virus", "Bakanae disease"],
            "severity": 6.5,
            "researchCount": 32,
            "trends": [
                {"year": "2020", "cases": 14},
                {"year": "2021", "cases": 16},
                {"year": "2022", "cases": 12},
                {"year": "2023", "cases": 10}
            ],
            "recommendations": [
                "Use certified seeds",
                "Treat seeds with fungicides",
                "Remove infected plants"
            ],
            "relatedResearch": [
                {"title": "Seed treatments for bakanae disease control", "authors": ["Siti R.", "Budi S."], "year": "2022"},
                {"title": "Virus resistance in Indonesian rice varieties", "authors": ["Dewi K.", "Agus P."], "year": "2021"}
            ]
        },
        {
            "id": "loc7",
            "name": "Dhaka, Bangladesh",
            "lat": 23.8103, 
            "lng": 90.4125,
            "diseases": ["Bacterial blight", "Sheath rot"],
            "severity": 7.8,
            "researchCount": 30,
            "trends": [
                {"year": "2020", "cases": 16},
                {"year": "2021", "cases": 20},
                {"year": "2022", "cases": 18},
                {"year": "2023", "cases": 15}
            ],
            "recommendations": [
                "Use resistant varieties",
                "Apply balanced fertilization",
                "Maintain field sanitation"
            ],
            "relatedResearch": [
                {"title": "Bacterial blight management in Bangladesh", "authors": ["Rahman M.", "Islam K."], "year": "2022"},
                {"title": "Sheath rot epidemiology in rice fields", "authors": ["Hossain A.", "Ahmed S."], "year": "2021"}
            ]
        },
        {
            "id": "loc8",
            "name": "Beijing, China",
            "lat": 39.9042, 
            "lng": 116.4074,
            "diseases": ["Rice blast", "Bacterial leaf blight"],
            "severity": 6.2,
            "researchCount": 65,
            "trends": [
                {"year": "2020", "cases": 28},
                {"year": "2021", "cases": 25},
                {"year": "2022", "cases": 20},
                {"year": "2023", "cases": 15}
            ],
            "recommendations": [
                "Use resistant varieties",
                "Apply silicon fertilizers",
                "Implement integrated disease management"
            ],
            "relatedResearch": [
                {"title": "Genetic diversity of rice blast pathogens in China", "authors": ["Li W.", "Zhang Y."], "year": "2022"},
                {"title": "Silicon-mediated resistance against rice blast", "authors": ["Wang H.", "Chen X."], "year": "2021"}
            ]
        },
        {
            "id": "loc9",
            "name": "Tokyo, Japan",
            "lat": 35.6762, 
            "lng": 139.6503,
            "diseases": ["Bacterial leaf blight", "Brown spot"],
            "severity": 5.5,
            "researchCount": 48,
            "trends": [
                {"year": "2020", "cases": 15},
                {"year": "2021", "cases": 12},
                {"year": "2022", "cases": 10},
                {"year": "2023", "cases": 8}
            ],
            "recommendations": [
                "Use resistant varieties",
                "Apply balanced fertilization",
                "Implement precision agriculture"
            ],
            "relatedResearch": [
                {"title": "Smart farming for rice disease management in Japan", "authors": ["Tanaka S.", "Ito M."], "year": "2022"},
                {"title": "Climate change impact on rice diseases in Japan", "authors": ["Yamamoto K.", "Suzuki T."], "year": "2021"}
            ]
        },
        {
            "id": "loc10",
            "name": "Los Baños, Philippines",
            "lat": 14.1652, 
            "lng": 121.2417,
            "diseases": ["Tungro virus", "Sheath blight"],
            "severity": 7.0,
            "researchCount": 52,
            "trends": [
                {"year": "2020", "cases": 22},
                {"year": "2021", "cases": 20},
                {"year": "2022", "cases": 18},
                {"year": "2023", "cases": 15}
            ],
            "recommendations": [
                "Use resistant varieties",
                "Control insect vectors",
                "Apply integrated pest management"
            ],
            "relatedResearch": [
                {"title": "IRRI's research on tungro virus resistance", "authors": ["Santos R.", "Cruz M."], "year": "2022"},
                {"title": "Vector management for tungro control", "authors": ["Perez J.", "Reyes L."], "year": "2021"}
            ]
        }
    ]
    
    return [ResearchLocation(**location) for location in locations]

@app.get("/disease-solutions", response_model=Dict[str, DiseaseSolution])
async def get_disease_solutions():
    """
    Get information about rice diseases and their solutions.
    """
    solutions = {
        "Rice Blast (โรคไหม้)": {
            "symptoms": "เริ่มจากจุดสีน้ำตาลเล็กๆ บนใบ ขยายเป็นแผลรูปตา มีขอบสีน้ำตาลเข้มและกลางแผลสีเทา ในระยะรุนแรงใบจะแห้งตาย",
            "solutions": [
                "ใช้พันธุ์ข้าวต้านทานโรค เช่น กข7, กข21, สุพรรณบุรี1",
                "ฉีดพ่นสารป้องกันกำจัดเชื้อรา เช่น ไตรไซคลาโซล คาร์เบนดาซิม",
                "ใส่ปุ๋ยไนโตรเจนในปริมาณที่เหมาะสม ไม่มากเกินไป",
                "กำจัดวัชพืชในนาข้าวและบริเวณใกล้เคียง"
            ],
            "prevention": [
                "ใช้เมล็ดพันธุ์ที่สะอาดปราศจากโรค",
                "ไม่ปลูกข้าวหนาแน่นเกินไป",
                "หลีกเลี่ยงการให้ปุ๋ยไนโตรเจนมากเกินไป",
                "ทำลายตอซังและฟางข้าวหลังการเก็บเกี่ยว"
            ]
        },
        "Bacterial Blight (โรคขอบใบแห้ง)": {
            "symptoms": "ใบมีรอยแผลสีเหลืองตามขอบใบ ต่อมาแผลจะแห้งเป็นสีน้ำตาล ในระยะรุนแรงทั้งใบจะแห้งตาย",
            "solutions": [
                "ใช้พันธุ์ข้าวต้านทานโรค เช่น กข7, สุพรรณบุรี60",
                "ฉีดพ่นสารประกอบทองแดง เช่น คอปเปอร์ไฮดรอกไซด์",
                "ระบายน้ำออกจากแปลงนาเมื่อพบการระบาด",
                "ใส่ปุ๋ยโพแทสเซียมเพื่อเพิ่มความต้านทานให้ต้นข้าว"
            ],
            "prevention": [
                "ใช้เมล็ดพันธุ์ที่สะอาดปราศจากโรค",
                "แช่เมล็ดในน้ำร้อน 52-54 องศาเซลเซียส นาน 30 นาที",
                "ไม่ใส่ปุ๋ยไนโตรเจนมากเกินไป",
                "กำจัดวัชพืชที่เป็นพืชอาศัยของเชื้อโรค"
            ]
        },
        "Brown Spot (โรคใบจุดสีน้ำตาล)": {
            "symptoms": "เกิดจุดสีน้ำตาลรูปไข่บนใบ มีขนาด 2-5 มิลลิเมตร มีขอบสีเหลือง พบมากในระยะข้าวออกรวง",
            "solutions": [
                "ใช้พันธุ์ข้าวต้านทานโรค",
                "ฉีดพ่นสารป้องกันกำจัดเชื้อรา เช่น แมนโคเซบ โพรพิโคนาโซล",
                "ใส่ปุ๋ยให้สมดุล โดยเฉพาะธาตุโพแทสเซียม",
                "ปรับสภาพดินให้เหมาะสม ไม่ให้เป็นกรดจัด"
            ],
            "prevention": [
                "ใช้เมล็ดพันธุ์ที่สะอาดปราศจากโรค",
                "แช่เมล็ดในสารป้องกันกำจัดเชื้อราก่อนปลูก",
                "ปรับปรุงดินด้วยปูนขาวหากดินเป็นกรด",
                "ใส่ปุ๋ยอินทรีย์เพื่อปรับปรุงโครงสร้างดิน"
            ]
        },
        "Sheath Blight (โรคกาบใบแห้ง)": {
            "symptoms": "เริ่มจากแผลรูปไข่สีเขียวอมเทาบนกาบใบ ต่อมาแผลขยายเป็นรูปไม่แน่นอน ขอบแผลสีน้ำตาลเข้ม ทำให้กาบใบแห้ง",
            "solutions": [
                "ฉีดพ่นสารป้องกันกำจัดเชื้อรา เช่น วาลิดามัยซิน ฟลูโทลานิล",
                "ลดความหนาแน่นของการปลูกข้าว",
                "ระบายน้ำออกจากแปลงนาเมื่อพบการระบาด",
                "ใส่ปุ๋ยไนโตรเจนในปริมาณที่เหมาะสม"
            ],
            "prevention": [
                "ไม่ปลูกข้าวหนาแน่นเกินไป",
                "กำจัดวัชพืชในนาข้าว",
                "ทำลายตอซังและฟางข้าวหลังการเก็บเกี่ยว",
                "ปลูกพืชหมุนเวียนเพื่อตัดวงจรโรค"
            ]
        },
        "Rice Tungro Virus (โรคใบสีส้ม)": {
            "symptoms": "ใบมีสีเหลืองส้ม ต้นแคระแกร็น ใบตั้งชัน ออกรวงน้อย เมล็ดลีบ",
            "solutions": [
                "ใช้พันธุ์ข้าวต้านทานโรค",
                "กำจัดแมลงพาหะ (เพลี้ยจักจั่นสีเขียว) ด้วยสารฆ่าแมลง",
                "ถอนและทำลายต้นที่เป็นโรค",
                "ปลูกข้าวพร้อมกันในพื้นที่เดียวกัน"
            ],
            "prevention": [
                "ใช้เมล็ดพันธุ์ที่สะอาดปราศจากโรค",
                "หลีกเลี่ยงการปลูกข้าวต่างเวลากันในพื้นที่ใกล้เคียง",
                "กำจัดวัชพืชที่เป็นพืชอาศัยของเชื้อไวรัสและแมลงพาหะ",
                "ตรวจแปลงนาอย่างสม่ำเสมอเพื่อพบการระบาดได้เร็ว"
            ]
        },
        "False Smut (โรคดอกกระถิน)": {
            "symptoms": "เกิดก้อนเชื้อราสีเขียวมะกอกถึงสีดำบนรวงข้าว ทำให้เมล็ดเสียหาย",
            "solutions": [
                "ฉีดพ่นสารป้องกันกำจัดเชื้อรา เช่น โพรพิโคนาโซล ไดฟีโนโคนาโซล",
                "ฉีดพ่นในระยะข้าวตั้งท้องถึงออกดอก",
                "ลดการใช้ปุ๋ยไนโตรเจนที่มากเกินไป",
                "เก็บเกี่ยวเมื่อข้าวสุกแก่พอดี"
            ],
            "prevention": [
                "ใช้เมล็ดพันธุ์ที่สะอาดปราศจากโรค",
                "ไม่ปลูกข้าวหนาแน่นเกินไป",
                "ใส่ปุ๋ยให้สมดุล",
                "ทำลายตอซังและฟางข้าวหลังการเก็บเกี่ยว"
            ]
        },
        "Bakanae Disease (โรคถอดฝักดาบ)": {
            "symptoms": "ต้นข้าวสูงผิดปกติ ใบเรียวยาวสีเขียวซีด ต่อมาจะเหลืองและแห้งตาย รากเน่า",
            "solutions": [
                "แช่เมล็ดในสารป้องกันกำจัดเชื้อราก่อนปลูก เช่น คาร์เบนดาซิม",
                "ถอนและทำลายต้นที่เป็นโรค",
                "ใช้สารชีวภัณฑ์ เช่น เชื้อราไตรโคเดอร์มา",
                "ปรับสภาพดินให้เหมาะสม"
            ],
            "prevention": [
                "ใช้เมล็ดพันธุ์ที่สะอาดปราศจากโรค",
                "แช่เมล็ดในน้ำร้อน 52-54 องศาเซลเซียส นาน 10 นาที",
                "ตากดินก่อนปลูก",
                "ทำความสะอาดเครื่องมือและอุปกรณ์การเกษตร"
            ]
        }
    }
    
    return {k: DiseaseSolution(**v) for k, v in solutions.items()}

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
        cluster_summary = pd.read_csv('C:\\Users\\msu65\\Desktop\\myResearch\\my_project_fastapi\\rice_disease_analysis\\cluster_summaries.csv')

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
            "คุณเป็นผู้เชี่ยวชาญระดับสูงในการวิจัยโรคข้าวและการวิเคราะห์ทางการเกษตร "
            "ข้อมูลที่ให้มาประกอบด้วยการวิเคราะห์การจัดกลุ่มบทความเกี่ยวกับโรคข้าวอย่างละเอียด "
            "ซึ่งเน้นการระบุโรคข้าวที่พบบ่อยที่สุดพร้อมผลกระทบ "
            "การรวบรวมวิธีการรักษาที่มีประสิทธิภาพพร้อมอัตราความสำเร็จที่เกี่ยวข้อง "
            "และการค้นพบข้อมูลเชิงลึกและแนวโน้มใหม่ๆ ในการวิจัยโรคข้าว "
            "นอกจากนี้ โปรดร่างโครงสร้างสำหรับการพัฒนาฐานข้อมูลที่ครอบคลุมซึ่งจัดหมวดหมู่โรคข้าว ผลกระทบ และการรักษา "
            "จากชุดข้อมูลที่ครอบคลุมนี้ ให้สรุปเชิงลึกที่มีรายละเอียดพร้อมคำแนะนำที่นำไปปฏิบัติได้สำหรับการวิจัยในอนาคตและการประยุกต์ใช้ในการจัดการโรคข้าว "
            "โปรดตรวจสอบให้แน่ใจว่าบทสรุปมีรายละเอียดและครอบคลุมทุกแง่มุมของข้อมูลที่ให้มา รวมถึงโรคข้าวที่พบบ่อยที่สุด ผลกระทบ การรักษาที่มีประสิทธิภาพ อาการทั่วไป ผลกระทบที่สังเกตได้ และแนวโน้มที่กำลังเกิดขึ้น "
            "นอกจากนี้ ให้วิเคราะห์คำแนะนำที่นำไปปฏิบัติได้อย่างละเอียด โดยเน้นความสำคัญของการจัดลำดับความสำคัญของการวิจัยเกี่ยวกับโรคที่มีผลกระทบสูง การส่งเสริมแนวทางการจัดการโรคอย่างยั่งยืน การพัฒนาเครื่องมือตรวจจับโรคแต่เนิ่นๆ การเพิ่มการศึกษาและการเข้าถึงเกษตรกร และการสร้างฐานข้อมูลโรคข้าวที่ครอบคลุม "
            "คำตอบของคุณควรมีความครอบคลุมและให้แผนงานที่ชัดเจนสำหรับการวิจัยในอนาคตและการประยุกต์ใช้ในการจัดการโรคข้าว\n\n"
            + message_content
        )

        # Call the DeepSeek API to summarize the output
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญระดับสูงในการวิจัยโรคข้าวและการวิเคราะห์ทางการเกษตร"},
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

