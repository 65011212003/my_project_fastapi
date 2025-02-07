import os
import time
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from Bio import Entrez
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import MiniBatchKMeans
import spacy
from spacy.matcher import PhraseMatcher
from gensim import corpora, models

# Rice Disease Analysis Research Configuration
Entrez.email = "65011212003@msu.ac.th"
Entrez.api_key = "250b38811eabf58300fe369fa32371342308"
nlp = spacy.load("en_core_web_sm")
logging.basicConfig(level=logging.INFO)

# Download NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define rice disease-specific terms
disease_terms = [
    "rice blast", "bacterial blight", "rice stripe virus", 
    "brown spot", "sheath blight"
]

treatment_terms = [
    "fungicide", "pesticide", "integrated pest management",
    "biocontrol", "chemical control", "cultural practices"
]

symptom_terms = [
    "leaf spotting", "wilting", "stunted growth", 
    "chlorosis", "necrosis", "lesions"
]

# Create a PhraseMatcher and add patterns for rice disease analysis
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("DISEASE", [nlp.make_doc(term) for term in disease_terms])
matcher.add("TREATMENT", [nlp.make_doc(term) for term in treatment_terms])
matcher.add("SYMPTOM", [nlp.make_doc(term) for term in symptom_terms])

def fetch_pubmed_data():
    cache_file = "pubmed_cache_10k.pkl"
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    search_terms = [
        "rice diseases", 
        "plant pathology rice", 
        "rice blast", 
        "bacterial blight rice",
        "rice virus diseases",
        "rice disease treatment"
    ]
    
    query = " OR ".join([f'"{term}"' for term in search_terms])
    
    # Get total count and webenv for large dataset
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0, usehistory="y")
    record = Entrez.read(handle)
    total_count = int(record["Count"])
    webenv = record["WebEnv"]
    query_key = record["QueryKey"]
    
    articles = []
    batch_size = 500  # Increased batch size
    retries = 5
    timeout = 60
    max_articles = min(1000, total_count)
    
    for start in range(0, max_articles, batch_size):
        for attempt in range(1, retries + 1):
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    retstart=start,
                    retmax=batch_size,
                    webenv=webenv,
                    query_key=query_key,
                    retmode="xml",
                    timeout=timeout
                )
                article_data = handle.read()
                soup = BeautifulSoup(article_data, "lxml-xml")
                
                with ThreadPoolExecutor(max_workers=8) as executor:
                    batch_results = list(executor.map(parse_article, soup.find_all('PubmedArticle')))
                
                articles.extend(batch_results)
                logging.info(f"Fetched articles from {start} to {start + batch_size}")
                break  # Exit the retry loop on success
            except Exception as e:
                logging.error(f"Attempt {attempt} for batch starting at {start} failed: {e}")
                time.sleep(2 ** attempt)
        else:
            logging.error(f"All {retries} attempts failed for batch starting at {start}. Skipping this batch.")
    
    df = pd.DataFrame(articles)
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    return df

def parse_article(article):
    """
    Extracts title, abstract, and PMID from a PubMedArticle.
    """
    abstract_tag = article.find('AbstractText')
    title_tag = article.find('ArticleTitle')
    pmid_tag = article.find('PMID')
    return {
        "title": title_tag.text if title_tag else "No Title",
        "abstract": abstract_tag.text if abstract_tag else "",
        "pmid": pmid_tag.text if pmid_tag else ""
    }

def preprocess_text(text):
    """
    Converts text to lowercase, tokenizes, and lemmatizes while removing stopwords.
    """
    doc = nlp(text.lower())
    sw = set(stopwords.words('english'))
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in sw]
    return ' '.join(tokens)

def cluster_articles(df):
    """
    Vectorizes the processed text using HashingVectorizer and applies k-means clustering to group rice disease articles.
    """
    vectorizer = HashingVectorizer(n_features=2048, stop_words='english')
    X = vectorizer.fit_transform(df['processed_abstract'])
    
    # Fixed number of clusters (10)
    kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=1000)
    df['cluster'] = kmeans.fit_predict(X)
    return df

def extract_entities(text):
    """
    Extracts diseases, treatments, and symptoms from the given text using both phrase matching
    and spaCy's NER to boost extraction performance, optimized for rice disease research.
    """
    doc = nlp(text)
    extracted = {"DISEASE": set(), "TREATMENT": set(), "SYMPTOM": set()}
    
    # Primary extraction using PhraseMatcher
    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        extracted[label].add(span.text.strip())
    
    # Additional extraction using spaCy's NER to enhance recall and improve scores
    for ent in doc.ents:
        ent_text = ent.text.strip().lower()
        if any(term in ent_text for term in disease_terms):
            extracted["DISEASE"].add(ent.text.strip())
        if any(term in ent_text for term in treatment_terms):
            extracted["TREATMENT"].add(ent.text.strip())
        if any(term in ent_text for term in symptom_terms):
            extracted["SYMPTOM"].add(ent.text.strip())
            
    # Convert sets to lists for easier use
    return {k: list(v) for k, v in extracted.items()}

def extract_info(df):
    """
    Gathers cluster-specific metrics including common rice disease entities and LDA topics.
    """
    results = []
    for cluster_num in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_num]
        texts = cluster_df['processed_abstract'].tolist()
        
        # Extract entities using the enhanced extraction technique
        entities = [extract_entities(text) for text in texts]
        
        # Efficient topic modeling: tokenize texts before building the dictionary and corpus
        tokenized_texts = [word_tokenize(text) for text in texts]
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]
        
        # Increase LDA passes for more coherent topics
        lda = models.LdaMulticore(corpus, num_topics=3, id2word=dictionary, workers=4, passes=20)
        
        results.append({
            "cluster": cluster_num,
            "size": len(cluster_df),
            "common_diseases": pd.Series([d for e in entities for d in e.get('DISEASE', [])]).value_counts().head(5).to_dict(),
            "common_treatments": pd.Series([t for e in entities for t in e.get('TREATMENT', [])]).value_counts().head(5).to_dict(),
            "common_symptoms": pd.Series([s for e in entities for s in e.get('SYMPTOM', [])]).value_counts().head(5).to_dict(),
            "topics": lda.print_topics()
        })
    return pd.DataFrame(results)

def compute_f1(true_set, pred_set):
    """
    Computes precision, recall, and F1 score given the true and predicted sets.
    Normalizes entities to lower-case and strips surrounding whitespace to get a fair comparison.
    """
    true_set = {x.lower().strip() for x in true_set}
    pred_set = {x.lower().strip() for x in pred_set}
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def compute_jaccard(true_set, pred_set):
    """
    Computes the Jaccard similarity between true and predicted sets after normalization.
    """
    true_set = {x.lower().strip() for x in true_set}
    pred_set = {x.lower().strip() for x in pred_set}
    union = true_set | pred_set
    if len(union) == 0:
        return 1.0  # Perfect match when both are empty.
    return len(true_set & pred_set) / len(union)

def compute_exact_match(true_set, pred_set):
    """
    Returns 1 if the predicted set exactly matches the true set (after normalization), otherwise 0.
    """
    return 1 if {x.lower().strip() for x in true_set} == {x.lower().strip() for x in pred_set} else 0

def evaluate_entity_extraction(test_samples):
    """
    Evaluates entity extraction performance using precision, recall, F1 score, Jaccard similarity, and exact match ratio.
    """
    scores = {
        "DISEASE": [],
        "TREATMENT": [],
        "SYMPTOM": []
    }
    
    for sample in test_samples:
        text = sample['text']
        true_entities = sample['true_entities']
        predicted_entities = extract_entities(text)
        
        for key in scores.keys():
            precision, recall, f1 = compute_f1(true_entities.get(key, []), predicted_entities.get(key, []))
            jaccard = compute_jaccard(true_entities.get(key, []), predicted_entities.get(key, []))
            exact_match = compute_exact_match(true_entities.get(key, []), predicted_entities.get(key, []))
            scores[key].append({
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": jaccard,
                "exact_match": exact_match
            })
    
    avg_scores = {}
    for key, score_list in scores.items():
        avg_precision = np.mean([s["precision"] for s in score_list])
        avg_recall = np.mean([s["recall"] for s in score_list])
        avg_f1 = np.mean([s["f1"] for s in score_list])
        avg_jaccard = np.mean([s["jaccard"] for s in score_list])
        avg_exact_match = np.mean([s["exact_match"] for s in score_list])
        avg_scores[key] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "jaccard": avg_jaccard,
            "exact_match": avg_exact_match
        }
    
    return avg_scores

def run_evaluation():
    test_samples = [
        {
            "text": "Rice blast is a devastating disease that affects rice crops. Application of fungicide can help control the outbreak.",
            "true_entities": {
                "DISEASE": ["rice blast"],
                "TREATMENT": ["fungicide"],
                "SYMPTOM": []
            }
        },
        {
            "text": "The common symptoms include leaf spotting and wilting in affected plants.",
            "true_entities": {
                "DISEASE": [],
                "TREATMENT": [],
                "SYMPTOM": ["leaf spotting", "wilting"]
            }
        },
        {
            "text": "Bacterial blight in rice can lead to necrosis and lesions, and integrated pest management is recommended.",
            "true_entities": {
                "DISEASE": ["bacterial blight"],
                "TREATMENT": ["integrated pest management"],
                "SYMPTOM": ["necrosis", "lesions"]
            }
        }
    ]
    
    avg_scores = evaluate_entity_extraction(test_samples)
    print("\nEntity Extraction Evaluation (Average Scores):")
    for entity_type, scores in avg_scores.items():
        print(f"\n{entity_type}:")
        print(f"  Precision:   {scores['precision']:.2f}")
        print(f"  Recall:      {scores['recall']:.2f}")
        print(f"  F1 Score:    {scores['f1']:.2f}")
        print(f"  Jaccard:     {scores['jaccard']:.2f}")
        print(f"  Exact Match: {scores['exact_match']:.2f}")

def main():
    logging.info("Starting data collection for rice disease research...")
    df = fetch_pubmed_data()
    
    if df.empty:
        logging.warning("No articles fetched. Exiting.")
        return

    logging.info("Preprocessing data...")
    df = df[df['abstract'].str.len() > 100].drop_duplicates(subset=['abstract'])
    df['processed_abstract'] = df['abstract'].apply(preprocess_text)
    
    logging.info("Clustering articles with optimal parameters for best score...")
    clustered_df = cluster_articles(df)
    
    logging.info("Extracting information...")
    results_df = extract_info(clustered_df)
    
    logging.info("Research Findings:")
    for _, row in results_df.iterrows():
        print(f"\nCluster {row['cluster']} ({row['size']} articles)")
        print("Common Diseases:", row['common_diseases'])
        print("Common Treatments:", row['common_treatments'])
        print("Common Symptoms:", row['common_symptoms'])
        print("Main Topics:")
        for topic in row['topics']:
            print(f"  Topic {topic[0]}: {topic[1]}")
    
    # Run entity extraction evaluation using various metrics.
    run_evaluation()

if __name__ == "__main__":
    main()
