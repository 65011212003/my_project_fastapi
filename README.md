# Rice Disease Analysis API

A FastAPI application for analyzing rice disease research articles from PubMed. This application provides both standard and advanced analysis capabilities for rice disease research.

## Features

### Standard Analysis
- Fetch articles from PubMed based on a query
- Process abstracts using NLP techniques
- Perform clustering and topic modeling
- Extract entities related to diseases and symptoms
- Visualize results with statistics and charts

### Advanced Analysis (New)
- Enhanced PubMed data collection with comprehensive search queries
- Domain-specific text preprocessing
- Advanced entity extraction for diseases, treatments, symptoms, and effects
- State-of-the-art embeddings using SciBERT or SentenceTransformer
- Dimensionality reduction with UMAP or LDA
- Improved clustering with KMeans or DBSCAN
- Detailed cluster analysis with entity frequencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rice-disease-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK and spaCy data:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the API server:
```bash
uvicorn main:app --reload
```

2. Access the API documentation at http://localhost:8000/docs

## API Endpoints

### Standard Analysis
- `POST /analyze`: Start the standard analysis process
- `GET /status`: Check the status of the analysis
- `GET /clusters`: Get information about clusters
- `GET /articles/{cluster_id}`: Get articles in a specific cluster
- `GET /topics`: Get LDA topics
- `GET /statistics`: Get statistical information
- `GET /articles`: Get paginated articles

### Advanced Analysis
- `POST /advanced-analyze`: Start the advanced analysis process
- `GET /advanced-status`: Check the status of the advanced analysis
- `GET /advanced-articles/{cluster_id}`: Get articles in a specific cluster from advanced analysis
- `GET /advanced-articles`: Get paginated articles from advanced analysis
- `GET /advanced-statistics`: Get detailed statistical information from advanced analysis

## Advanced Analysis Options

The advanced analysis endpoint accepts the following parameters:

- `query`: Optional custom search query (if not provided, a comprehensive query is built)
- `max_results`: Maximum number of articles to fetch (default: 2000)
- `use_cache`: Whether to use cached data if available (default: true)
- `embedding_method`: Method for generating embeddings ("scibert", "sentence_transformer", or "auto")
- `clustering_method`: Method for clustering ("kmeans" or "dbscan")
- `dimension_reduction`: Method for dimensionality reduction ("umap", "lda", or "auto")

## Dependencies

- FastAPI: Web framework
- Biopython: For PubMed data retrieval
- NLTK & spaCy: For NLP processing
- scikit-learn: For clustering and vectorization
- Transformers & SentenceTransformers: For advanced embeddings
- UMAP: For dimensionality reduction
- BeautifulSoup: For XML parsing
- Pandas & NumPy: For data manipulation
- Matplotlib & Seaborn: For visualization

## License

[MIT License](LICENSE) 