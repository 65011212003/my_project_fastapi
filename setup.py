import nltk
import subprocess
import sys

def setup():
    # Download required NLTK data
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Install spaCy English model
    print("Installing spaCy English model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

if __name__ == "__main__":
    setup() 