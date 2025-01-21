import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
from nltk.corpus import wordnet as wn
from urllib.parse import quote_plus
import time

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
        for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            except Exception as e:
                st.error(f"Error downloading {resource}: {str(e)}")
                return False
        return True
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        return False

class ContentOptimizer:
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

    def analyze_competitor_content(self, content):
        """ Analyze competitor content based on structure, tone, depth, and keywords """
        # Return a structured analysis
        analysis = {
            'headings': self.extract_headings(content),
            'tone': self.analyze_tone(content),
            'depth': self.analyze_depth(content),
            'keyword_relevance': self.analyze_keywords(content)
        }
        return analysis
    
    def extract_headings(self, content):
        """ Extract and analyze headings and subheadings """
        headings = re.findall(r'(^|\n)[#]{1,6}\s*(.*?)\s*(?=\n|$)', content)
        return headings
    
    def analyze_tone(self, content):
        """ Simple tone analysis based on keyword matching """
        tone_keywords = {
            'professional': ['research', 'analysis', 'methodology'],
            'conversational': ['you', 'we', 'let\'s', 'try'],
            'promotional': ['buy', 'discount', 'offers']
        }
        
        tone_scores = {'professional': 0, 'conversational': 0, 'promotional': 0}
        
        for tone, keywords in tone_keywords.items():
            tone_scores[tone] = sum([content.lower().count(keyword) for keyword in keywords])
        
        # Determine dominant tone
        dominant_tone = max(tone_scores, key=tone_scores.get)
        return dominant_tone
    
    def analyze_depth(self, content):
        """ Measure content depth based on sentence length and complexity """
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_length > 20:
            return "High"
        elif avg_sentence_length > 10:
            return "Medium"
        else:
            return "Low"

    def analyze_keywords(self, content):
        """ Check keyword usage and integration """
        keywords = ['qualifying r&d expenditure', 'tax relief', 'government incentives']  # Example keywords
        keyword_count = {keyword: content.lower().count(keyword) for keyword in keywords}
        return keyword_count
    
    def get_semantic_keywords(self, text):
        """ Generate related words using synonyms from WordNet """
        words = self.tokenize_text(text)
        related_keywords = set()
        
        # Generate synonyms using WordNet
        for word in words:
            if word not in self.stop_words:
                synonyms = set()
                for syn in wn.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name())
                related_keywords.update(synonyms)
        
        # Limit the number of keywords to 10
        return list(related_keywords)[:10]

    def get_competitor_urls_from_serp(self, keyword):
        """ Scrape Google SERP for the top 5 competitor URLs based on the keyword """
        headers = {'User-Agent': 'Mozilla/5.0'}
        query = quote_plus(keyword)
        search_url = f'https://www.google.com/search?q={query}'
        
        try:
            response = requests.get(search_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [a['href'] for a in soup.find_all('a', href=True)]
            
            # Filter out URLs and limit to top 5
            competitor_urls = []
            for link in links:
                if link.startswith('/url?q='):
                    competitor_urls.append(link.split('/url?q=')[1].split('&')[0])
                    if len(competitor_urls) == 5:
                        break
            return competitor_urls
        except Exception as e:
            st.error(f"Error fetching Google SERP: {str(e)}")
            return []

def main():
    st.set_page_config(page_title="Content Analysis & Optimization", layout="wide")
    
    st.title("Content Analysis & Optimization Tool for Featured Snippet Success")
    
    # Initialize NLTK (but don't block if it fails)
    setup_nltk()
    
    # Initialize analyzer
    optimizer = ContentOptimizer()
    
    # Create two columns for competitor and user content
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("Competitor Content (Featured Snippet Holder)")
        competitor_content = st.text_area("Enter the competitor's content here:", height=400)
        
        st.subheader("User Content (Your Content)")
        user_content = st.text_area("Enter your content here:", height=400)
        
        target_keyword = st.text_input("Enter target keyword:")
        analyze_button = st.button("Analyze Content")
        
    with col2:
        st.subheader("Analysis Results")
        
        if analyze_button and competitor_content and user_content:
            try:
                # Analyze competitor content
                competitor_analysis = optimizer.analyze_competitor_content(competitor_content)
                
                # Display competitor strengths
                st.subheader("Competitor Strengths")
                st.write(f"**Headings Structure:** {competitor_analysis['headings']}")
                st.write(f"**Tone:** {competitor_analysis['tone']}")
                st.write(f"**Content Depth:** {competitor_analysis['depth']}")
                st.write(f"**Keyword Relevance:** {competitor_analysis['keyword_relevance']}")
                
                # Analyze user content
                user_analysis = optimizer.analyze_competitor_content(user_content)
                
                # Display user weaknesses
                st.subheader("User Content Weaknesses")
                st.write(f"**Headings Structure:** {user_analysis['headings']}")
                st.write(f"**Tone:** {user_analysis['tone']}")
                st.write(f"**Content Depth:** {user_analysis['depth']}")
                st.write(f"**Keyword Relevance:** {user_analysis['keyword_relevance']}")
                
                # Get semantic keyword suggestions
                semantic_keywords = optimizer.get_semantic_keywords(user_content)
                if semantic_keywords:
                    st.subheader("Semantic Keyword Suggestions")
                    st.write("Consider adding these related keywords to enhance your content:")
                    for keyword in semantic_keywords:
                        st.markdown(f"- {keyword}")
                    
                # Get competitor URLs from Google SERP
                if target_keyword.strip():
                    competitor_urls = optimizer.get_competitor_urls_from_serp(target_keyword)
                    
                    if competitor_urls:
                        st.subheader("Suggested Keywords from Competitors")
                        for url in competitor_urls:
                            st.write(url)
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please check your input and try again.")

if __name__ == "__main__":
    main()
