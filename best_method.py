import os
import time
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from Bio import Entrez
from bs4 import BeautifulSoup

import spacy
from spacy.matcher import PhraseMatcher

import nltk
from nltk.corpus import stopwords

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation

import umap

# ---------------------------
# Configuration and Setup
# ---------------------------
CONFIG = {
    'pubmed_cache_file': "pubmed_cache.pkl",
    'batch_size': 500,
    'max_articles': 2000,  # Increased to get more comprehensive data
    'retries': 5,
    'timeout': 60,
    'random_state': 42,
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Entrez configuration for PubMed
Entrez.email = "65011212003@msu.ac.th"
Entrez.api_key = "250b38811eabf58300fe369fa32371342308"

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ---------------------------
# Biomedical Language Model Setup
# ---------------------------
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    # Load SciBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
except ImportError:
    logging.warning("Transformers library not available. Please install it for biomedical embeddings.")
    tokenizer = None
    model = None

# Fallback to SentenceTransformer if transformers not available
if model is None:
    try:
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        logging.warning("SentenceTransformer is not available. Please install it if needed.")
        sentence_model = None

# ---------------------------
# Define Comprehensive Rice Disease Terms and PhraseMatcher
# ---------------------------
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

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("DISEASE", [nlp.make_doc(term) for term in disease_terms])
matcher.add("TREATMENT", [nlp.make_doc(term) for term in treatment_terms])
matcher.add("SYMPTOM", [nlp.make_doc(term) for term in symptom_terms])
matcher.add("EFFECT", [nlp.make_doc(term) for term in effects_terms])

# ---------------------------
# Data Fetching with Comprehensive Search
# ---------------------------
def build_search_query():
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

def parse_article(article):
    """Extracts title, abstract, PMID, and publication year from a PubMedArticle."""
    abstract_tag = article.find('AbstractText')
    title_tag = article.find('ArticleTitle')
    pmid_tag = article.find('PMID')
    year_tag = article.find('PubDate').find('Year')
    
    return {
        "title": title_tag.text if title_tag else "No Title",
        "abstract": abstract_tag.text if abstract_tag else "",
        "pmid": pmid_tag.text if pmid_tag else "",
        "year": year_tag.text if year_tag else ""
    }

def fetch_pubmed_data():
    """
    Fetches comprehensive PubMed data on rice diseases.
    Includes validation steps and error handling.
    """
    cache_file = CONFIG['pubmed_cache_file']
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        if not df.empty:
            logging.info(f"Loaded {len(df)} articles from cache")
            return df

    query = build_search_query()
    
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0, usehistory="y")
    record = Entrez.read(handle)
    total_count = int(record["Count"])
    webenv = record["WebEnv"]
    query_key = record["QueryKey"]

    logging.info(f"Found {total_count} articles matching the search criteria")

    articles = []
    for start in range(0, min(CONFIG['max_articles'], total_count), CONFIG['batch_size']):
        for attempt in range(1, CONFIG['retries'] + 1):
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    retstart=start,
                    retmax=CONFIG['batch_size'],
                    webenv=webenv,
                    query_key=query_key,
                    retmode="xml",
                    timeout=CONFIG['timeout']
                )
                article_data = handle.read()
                soup = BeautifulSoup(article_data, "lxml-xml")
                
                with ThreadPoolExecutor(max_workers=8) as executor:
                    batch_results = list(executor.map(parse_article, soup.find_all('PubmedArticle')))
                
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
        df = df[df['abstract'].str.len() > 100]  # Remove entries with very short abstracts
        df = df.drop_duplicates(subset=['abstract'])  # Remove duplicates
        
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        logging.info(f"Successfully collected and validated {len(df)} unique articles")
    else:
        logging.error("No valid articles were collected")
        
    return df

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_text_improved(text):
    """
    Enhanced text preprocessing with domain-specific considerations.
    """
    doc = nlp(text.lower())
    # Keep domain-specific terms intact while removing general stopwords
    tokens = [
        token.lemma_ for token in doc 
        if (token.is_alpha and 
            token.text not in stopwords.words('english')) or 
        token.text in disease_terms + treatment_terms + symptom_terms
    ]
    return " ".join(tokens)

# ---------------------------
# Vectorization & Clustering
# ---------------------------
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
    if method == 'umap':
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.1,
            random_state=CONFIG['random_state']
        )
        return reducer.fit_transform(X)
    elif method == 'lda':
        lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=CONFIG['random_state']
        )
        return lda.fit_transform(X)

def determine_optimal_clusters(X, max_clusters=10):
    """
    Determines optimal number of clusters using silhouette score.
    """
    scores = []
    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=CONFIG['random_state'])
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    
    optimal_clusters = scores.index(max(scores)) + 2
    logging.info(f"Optimal number of clusters determined: {optimal_clusters}")
    return optimal_clusters

def cluster_data(X, method='kmeans', n_clusters=None):
    """
    Clusters data using either KMeans or DBSCAN.
    """
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(X)
        
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=CONFIG['random_state'])
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
    
    labels = model.fit_predict(X)
    return labels, model

# ---------------------------
# Entity Extraction & Summarization
# ---------------------------
def extract_entities(text):
    """
    Extracts and categorizes entities related to rice diseases.
    """
    doc = nlp(text)
    matches = matcher(doc)
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

def summarize_clusters(df):
    """
    Provides comprehensive cluster summaries with entity frequencies.
    """
    results = []
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        texts = cluster_df['processed_abstract'].tolist()
        entities_list = [extract_entities(text) for text in texts]

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

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    logging.info("Starting comprehensive rice disease research analysis...")

    # 1) Data Collection
    df = fetch_pubmed_data()
    if df.empty:
        logging.error("No articles fetched. Exiting.")
        return

    # 2) Preprocessing
    df = df[df['abstract'].str.len() > 100].drop_duplicates(subset=['abstract'])
    df['processed_abstract'] = df['abstract'].apply(preprocess_text_improved)

    # 3) Vectorization using SciBERT
    texts = df['processed_abstract'].tolist()
    X_sent = get_scibert_embeddings(texts)
    if X_sent is None:
        logging.error("Could not obtain embeddings. Exiting.")
        return

    # 4) Dimensionality Reduction
    X_reduced = reduce_dimensions(X_sent, method='umap', n_components=50)

    # 5) Clustering
    labels, model = cluster_data(X_reduced, method='kmeans')
    df['cluster'] = labels

    # 6) Save Results
    df_reduced = pd.DataFrame(
        X_reduced,
        columns=[f"dimension_{i}" for i in range(X_reduced.shape[1])]
    )
    df_reduced.to_csv("reduced_features.csv", index=False)
    df.to_csv("clustered_data.csv", index=False)

    # 7) Cluster Analysis
    cluster_summary = summarize_clusters(df)
    cluster_summary.to_csv("cluster_summary.csv", index=False)

    # 8) Results Presentation
    for _, row in cluster_summary.iterrows():
        print(f"\nCluster {row['cluster']} ({row['size']} articles)")
        print("Major Diseases:", row.get('common_diseases', {}))
        print("Common Treatments:", row.get('common_treatments', {}))
        print("Typical Symptoms:", row.get('common_symptoms', {}))
        print("Observed Effects:", row.get('common_effects', {}))

    # 9) Validation Metrics
    if len(set(labels)) > 1:
        score = silhouette_score(X_reduced, labels)
        logging.info(f"Clustering Validation - Silhouette Score: {score:.4f}")

if __name__ == "__main__":
    main()
