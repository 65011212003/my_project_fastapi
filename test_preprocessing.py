from preprocessing import search_pubmed, fetch_details, preprocess_text, extract_entities_simple, save_preprocessing_steps_to_excel
from Bio import Entrez

# Set your email for Entrez
Entrez.email = "65011212003@msu.ac.th"  # Email for PubMed API access

def test_preprocessing():
    # Test with a small query
    print("Searching PubMed...")
    query = "rice blast disease[Title] AND (\"2023\"[Date - Publication])"
    pmids = search_pubmed(query, max_results=5)
    
    if pmids:
        print(f"Found {len(pmids)} articles")
        articles = fetch_details(pmids)
        
        # Save preprocessing steps to Excel
        excel_file = save_preprocessing_steps_to_excel(articles)
        print(f"\nPreprocessing steps saved to: {excel_file}")
        
        # Display summary of articles and entities
        for article in articles:
            print("\nArticle:")
            print(f"PMID: {article['PMID']}")
            print(f"Title: {article['Title']}")
            
            # Get preprocessing steps
            steps = preprocess_text(article['Abstract'])
            print("\nProcessed Abstract:")
            print(steps['final'][:200] + "...")  # Show first 200 chars
            
            print("\nNamed Entities:")
            entities = extract_entities_simple(article['Abstract'])
            for entity_type, values in entities.items():
                print(f"{entity_type}: {', '.join(values[:3])}")  # Show first 3 entities of each type
            
            print("-" * 80)
    else:
        print("No articles found")

if __name__ == "__main__":
    test_preprocessing() 