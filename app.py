import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Initialize NLTK
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        return True
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        return False

class SEOAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        return tokens
        
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
                    ngram_range=(1, 2)  # Include bigrams
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
        
        # Length score (30 points max)
        words = word_tokenize(content)
        if len(words) >= 1000:
            score_breakdown['length'] = 30
        elif len(words) >= 500:
            score_breakdown['length'] = 20
        else:
            score_breakdown['length'] = 10
        
        # Readability score (40 points max)
        try:
            sentences = sent_tokenize(content)
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            if avg_sentence_length <= 20:
                score_breakdown['readability'] = 40
            elif avg_sentence_length <= 25:
                score_breakdown['readability'] = 30
            else:
                score_breakdown['readability'] = 20
        except:
            score_breakdown['readability'] = 20
        
        # Keyword density score (30 points max)
        try:
            tokens = self.preprocess_text(content)
            word_freq = Counter(tokens)
            total_words = len(tokens)
            
            # Calculate keyword density
            keyword_densities = {word: (count/total_words)*100 
                               for word, count in word_freq.most_common(10)}
            
            # Score based on keyword density (ideal: 1-3%)
            good_density_keywords = sum(1 for density in keyword_densities.values() 
                                     if 1 <= density <= 3)
            score_breakdown['keyword_density'] = min(30, good_density_keywords * 3)
        except:
            score_breakdown['keyword_density'] = 0
            
        return sum(score_breakdown.values()), score_breakdown, dict(word_freq.most_common(10))

def main():
    st.set_page_config(page_title="SEO Content Analyzer", layout="wide")
    
    st.title("SEO Content Analyzer")
    
    # Initialize NLTK
    if not setup_nltk():
        st.error("Failed to initialize NLTK. Please try again.")
        return
    
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
