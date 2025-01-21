import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
import sys

# Install NLTK and download required data
def setup_nltk():
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        return nltk
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        return None

# Install spaCy and download model
def setup_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load('en_core_web_sm')
        return nlp
    except Exception as e:
        st.error(f"Error setting up spaCy: {str(e)}")
        return None

# Initialize NLP components
@st.cache_resource
def initialize_nlp():
    nltk = setup_nltk()
    nlp = setup_spacy()
    return nltk, nlp

class SEOAnalyzer:
    def __init__(self, nltk, nlp):
        self.nltk = nltk
        self.nlp = nlp
        
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
                except:
                    continue
            
            if not all_text:
                return [], []
                
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_text)
            feature_names = vectorizer.get_feature_names_out()
            
            return feature_names, all_text
    
    def calculate_content_score(self, content, competitor_keywords):
        if not self.nltk:
            return 0, {'error': 'NLTK not available'}
            
        score_breakdown = {
            'length': 0,
            'keyword_usage': 0,
            'readability': 0
        }
        
        # Length score
        words = content.split()
        if len(words) >= 1000:
            score_breakdown['length'] = 30
        elif len(words) >= 500:
            score_breakdown['length'] = 20
        else:
            score_breakdown['length'] = 10
            
        # Keyword usage score
        if competitor_keywords:
            content_words = set(self.nltk.word_tokenize(content.lower()))
            keyword_matches = content_words.intersection(set(competitor_keywords))
            score_breakdown['keyword_usage'] = int((len(keyword_matches) / len(competitor_keywords)) * 40)
        
        # Readability score
        sentences = [s for s in content.split('.') if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        if avg_sentence_length <= 20:
            score_breakdown['readability'] = 30
        elif avg_sentence_length <= 25:
            score_breakdown['readability'] = 20
        else:
            score_breakdown['readability'] = 10
            
        total_score = sum(score_breakdown.values())
        return total_score, score_breakdown

def main():
    st.set_page_config(page_title="SEO Content Analyzer", layout="wide")
    
    st.title("SEO Content Analyzer")
    
    # Initialize NLP components
    nltk, nlp = initialize_nlp()
    
    if not nltk or not nlp:
        st.error("Failed to initialize required NLP components. Please check the logs.")
        return
    
    # Initialize analyzer
    analyzer = SEOAnalyzer(nltk, nlp)
    
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
                # Process competitor URLs
                urls = [url.strip() for url in competitor_urls.split('\n') if url.strip()]
                
                # Analyze competitor content
                competitor_keywords, competitor_texts = analyzer.analyze_competitor_content(urls)
                
                # Calculate scores
                score, score_breakdown = analyzer.calculate_content_score(content, competitor_keywords)
                
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
                if 'error' not in score_breakdown:
                    breakdown_df = pd.DataFrame({
                        'Category': score_breakdown.keys(),
                        'Score': score_breakdown.values()
                    })
                    st.bar_chart(breakdown_df.set_index('Category'))
                
                    # Display keyword suggestions
                    st.subheader("Keyword Suggestions")
                    if competitor_keywords:
                        content_words = set(nltk.word_tokenize(content.lower()))
                        missing_keywords = set(competitor_keywords) - content_words
                        
                        if missing_keywords:
                            st.write("Consider adding these keywords to your content:")
                            for keyword in missing_keywords:
                                st.markdown(f"- {keyword}")
                        else:
                            st.write("Great job! Your content covers most important keywords.")
                    else:
                        st.warning("No competitor keywords found. Please check the URLs provided.")
                else:
                    st.error("Error in analysis. Please try again.")
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please check your input and try again.")

if __name__ == "__main__":
    main()
