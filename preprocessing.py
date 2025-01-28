import re
import nltk
import spacy
import logging
import pandas as pd
from Bio import Entrez
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime

def search_pubmed(query, max_results=10000):
    """
    Search PubMed for articles matching the query.
    Returns a list of PMIDs.
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        logging.error(f"Error searching PubMed: {str(e)}")
        return []

def fetch_details(id_list):
    """
    Fetch article details from PubMed given a list of PMIDs.
    Returns a list of dictionaries containing article details.
    """
    articles = []
    try:
        ids = ','.join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="xml")
        records = Entrez.read(handle)['PubmedArticle']
        
        for record in records:
            try:
                # Extract required fields
                article_data = record['MedlineCitation']
                pmid = article_data['PMID']
                article = article_data['Article']
                title = article['ArticleTitle']
                
                # Handle abstract text which might be a list of XML elements
                abstract_element = article.get('Abstract', {}).get('AbstractText', [''])[0]
                if hasattr(abstract_element, 'attributes'):
                    abstract = str(abstract_element)
                else:
                    abstract = abstract_element
                
                # Get preprocessing steps
                preprocessing_steps = preprocess_text(abstract)
                
                articles.append({
                    "PMID": pmid,
                    "Title": title, 
                    "Abstract": abstract,
                    "Processed_Abstract": preprocessing_steps["final"],
                    "Preprocessing_Steps": preprocessing_steps
                })
                
            except Exception as e:
                logging.error(f"Error processing record: {str(e)}")
                continue
                
        handle.close()
        return articles
        
    except Exception as e:
        logging.error(f"Error fetching details: {str(e)}")
        return articles

def preprocess_text(text):
    """
    Preprocess text by performing the following steps and return all intermediate results
    """
    if not isinstance(text, str):
        return {
            "original": "",
            "lowercase": "",
            "no_special_chars": "",
            "tokens": [],
            "no_stopwords": [],
            "lemmatized": [],
            "no_short_words": [],
            "final": ""
        }
    
    # Store each step
    steps = {
        "original": text,
        "lowercase": text.lower(),
    }
    
    # Remove special characters and numbers
    steps["no_special_chars"] = re.sub(r'[^a-zA-Z\s]', '', steps["lowercase"])
    
    # Tokenization
    steps["tokens"] = word_tokenize(steps["no_special_chars"])
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    steps["no_stopwords"] = [token for token in steps["tokens"] if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    steps["lemmatized"] = [lemmatizer.lemmatize(token) for token in steps["no_stopwords"]]
    
    # Remove short words
    steps["no_short_words"] = [token for token in steps["lemmatized"] if len(token) > 2]
    
    # Final processed text
    steps["final"] = ' '.join(steps["no_short_words"])
    
    return steps

def save_preprocessing_steps_to_excel(articles, output_file=None):
    """
    Save preprocessing steps for each article to an Excel file
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"preprocessing_steps_{timestamp}.xlsx"
    
    # Create a list to store all preprocessing steps
    all_steps = []
    
    for article in articles:
        # Get preprocessing steps
        steps = preprocess_text(article["Abstract"])
        
        # Create a row for each step
        step_data = {
            "PMID": article["PMID"],
            "Title": article["Title"],
            "Original": steps["original"],
            "Lowercase": steps["lowercase"],
            "No Special Chars": steps["no_special_chars"],
            "Tokens": ", ".join(steps["tokens"]),
            "No Stopwords": ", ".join(steps["no_stopwords"]),
            "Lemmatized": ", ".join(steps["lemmatized"]),
            "No Short Words": ", ".join(steps["no_short_words"]),
            "Final": steps["final"]
        }
        
        all_steps.append(step_data)
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(all_steps)
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    return output_file

def extract_entities_simple(text):
    """
    Extract named entities from text using spaCy.
    Returns a dictionary of entity types and their values.
    """
    if not isinstance(text, str):
        # Handle XML element or other non-string types
        text = str(text)
        # Remove XML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
        
    return entities
