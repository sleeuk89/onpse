import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy
import os

# Set NLTK data path to a writable directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.insert(0, nltk_data_dir)

# Initialize NLTK
@st.cache_resource
def setup_nltk():
    try:
        # Download required NLTK data
        for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            except Exception as e:
                st.error(f"Error downloading {resource}: {str(e)}")
                return False
        return True
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        return False

# Load spaCy's English model for semantic analysis
nlp = spacy.load('en_core_web_md')  # You may need to install it with 'pip install spacy' and 'python -m spacy download en_core_web_md'

class SEOAnalyzer:
    def __init__(self):
        # Basic tokenization without relying on NLTK initially
        self.stop_words = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                             'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                             'to', 'was', 'were', 'will', 'with'])
    
    def tokenize_text(self, text):
        """Simple tokenization fallback"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Split into words
        return text.split()
    
    def analyze_competitor_content(self, urls):
        all_text = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        with st.spinner('Analyzing competitor content...'):
            for url in urls:
                try:
                    response = requests.get(url, headers=headers, timeout=5)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = ' '.join([p.text for p in soup.find_all('p')])
                    all_text.append(text)
                except Exception as e:
                    st.warning(f"Couldn't analyze {url}: {str(e)}")
                    continue
            
            if not all_text:
                return [], []
                
            try:
                vectorizer = TfidfVectorizer(
                    max_features=50,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                tfidf_matrix = vectorizer.fit_transform(all_text)
                feature_names = vectorizer.get_feature_names_out()
                return feature_names, all_text
            except Exception as e:
                st.error(f"Error in keyword extraction: {str(e)}")
                return [], all_text
    
    def calculate_content_score(self, content):
        score_breakdown = {
            'length': 0,
            'readability': 0,
            'keyword_density': 0
        }
        
        # Tokenize text
        words = self.tokenize_text(content)
        
        # Length score (30 points max)
        if len(words) >= 1000:
            score_breakdown['length'] = 30
        elif len(words) >= 500:
            score_breakdown['length'] = 20
        else:
            score_breakdown['length'] = 10
        
        # Readability score (40 points max)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        if avg_sentence_length <= 20:
            score_breakdown['readability'] = 40
        elif avg_sentence_length <= 25:
            score_breakdown['readability'] = 30
        else:
            score_breakdown['readability'] = 20
        
        # Keyword density score (30 points max)
        word_freq = Counter(words)
        total_words = len(words)
        
        # Remove stop words from frequency count
        word_freq = Counter({word: count for word, count in word_freq.items() 
                           if word not in self.stop_words and len(word) > 2})
        
        # Calculate keyword density
        keyword_densities = {word: (count/total_words)*100 
                           for word, count in word_freq.most_common(10)}
        
        # Score based on keyword density (ideal: 1-3%)
        good_density_keywords = sum(1 for density in keyword_densities.values() 
                                 if 1 <= density <= 3)
        score_breakdown['keyword_density'] = min(30, good_density_keywords * 3)
        
        return sum(score_breakdown.values()), score_breakdown, dict(word_freq.most_common(10))

    def get_semantic_keywords(self, text):
        doc = nlp(text)
        related_keywords = set()
        
        # Extract entities and related terms (synonyms, hypernyms, etc.)
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop:
                related_keywords.add(token.lemma_)
        
        # Use spaCy's word vectors to find similar terms
        for word in related_keywords:
            similar_words = nlp.vocab[word].similarity
            related_keywords.update([similar_word for similar_word in similar_words])
        
        return list(related_keywords)


def main():
    st.set_page_config(page_title="SEO Content Analyzer", layout="wide")
    
    st.title("SEO Content Analyzer")
    
    # Initialize NLTK (but don't block if it fails)
    setup_nltk()
    
    # Initialize analyzer
    analyzer = SEOAnalyzer()
    
    # Create two columns
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("Content Editor")
        content = st.text_area("Enter your content here:", height=300)
        
        target_keywords = st.text_input("Enter target keywords (comma-separated):")
        competitor_urls = st.text_area("Enter competitor URLs (one per line):", height=100)
        
        analyze_button = st.button("Analyze Content")
        
    with col2:
        st.subheader("Analysis Results")
        
        if analyze_button and content:
            try:
                # Calculate content scores
                score, score_breakdown, keyword_freq = analyzer.calculate_content_score(content)
                
                # Display score with gauge chart
                fig = px.pie(values=[score, 100-score], 
                            names=['Score', 'Remaining'],
                            hole=0.7,
                            color_discrete_sequence=['#00CC96', '#EFF3F6'])
                fig.update_layout(
                    annotations=[dict(text=f'{score}', x=0.5, y=0.5, font_size=40, showarrow=False)]
                )
                st.plotly_chart(fig)
                
                # Display score breakdown
                st.subheader("Score Breakdown")
                breakdown_df = pd.DataFrame({
                    'Category': score_breakdown.keys(),
                    'Score': score_breakdown.values()
                })
                st.bar_chart(breakdown_df.set_index('Category'))
                
                # Display keyword frequency
                st.subheader("Top Keywords in Your Content")
                keyword_df = pd.DataFrame(
                    {'Keyword': keyword_freq.keys(),
                     'Frequency': keyword_freq.values()}
                )
                st.table(keyword_df)
                
                # Get semantic keyword suggestions
                semantic_keywords = analyzer.get_semantic_keywords(content)
                if semantic_keywords:
                    st.subheader("Semantic Keyword Suggestions")
                    st.write("Consider adding these related keywords to enhance your content:")
                    for keyword in semantic_keywords:
                        st.markdown(f"- {keyword}")
                    
                # Analyze competitor content if URLs provided
                if competitor_urls.strip():
                    urls = [url.strip() for url in competitor_urls.split('\n') if url.strip()]
                    competitor_keywords, _ = analyzer.analyze_competitor_content(urls)
                    
                    if competitor_keywords:
                        st.subheader("Suggested Keywords from Competitors")
                        current_keywords = set(keyword_freq.keys())
                        missing_keywords = set(competitor_keywords) - current_keywords
                        
                        if missing_keywords:
                            st.write("Consider adding these keywords to your content:")
                            for keyword in missing_keywords:
                                st.markdown(f"- {keyword}")
                        else:
                            st.write("Great job! Your content covers most important keywords.")
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please check your input and try again.")

if __name__ == "__main__":
    main()
