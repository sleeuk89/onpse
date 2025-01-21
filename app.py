import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import wordnet as wn
import re
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
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def tokenize_text(self, text):
        """ Simple tokenization without punctuation """
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation
        return text.split()

    def analyze_content_structure(self, content):
        """ Analyze headings and subheadings structure """
        headings = re.findall(r'(^|\n)[#]{1,6}\s*(.*?)\s*(?=\n|$)', content)
        return headings

    def analyze_tone(self, content):
        """ Analyze tone based on keyword matching """
        tone_keywords = {
            'professional': ['research', 'analysis', 'methodology'],
            'conversational': ['you', 'we', 'let\'s', 'try'],
            'promotional': ['buy', 'discount', 'offers']
        }

        tone_scores = {'professional': 0, 'conversational': 0, 'promotional': 0}
        for tone, keywords in tone_keywords.items():
            tone_scores[tone] = sum([content.lower().count(keyword) for keyword in keywords])

        dominant_tone = max(tone_scores, key=tone_scores.get)
        return dominant_tone

    def analyze_depth(self, content):
        """ Measure content depth by sentence length """
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_length > 20:
            return "High"
        elif avg_sentence_length > 10:
            return "Medium"
        else:
            return "Low"

    def analyze_keywords(self, content, target_keywords):
        """ Check keyword usage in content """
        keyword_count = {keyword: content.lower().count(keyword) for keyword in target_keywords}
        return keyword_count

    def get_semantic_keywords(self, text):
        """ Generate related words using WordNet """
        words = self.tokenize_text(text)
        related_keywords = set()

        for word in words:
            if word not in self.stop_words:
                synonyms = set()
                for syn in wn.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name())
                related_keywords.update(synonyms)

        # Limit to 10 semantic keywords
        return list(related_keywords)[:10]

    def get_competitor_urls_from_serp(self, keyword):
        """ Scrape Google SERP for top competitor URLs based on keyword """
        headers = {'User-Agent': 'Mozilla/5.0'}
        query = quote_plus(keyword)
        search_url = f'https://www.google.com/search?q={query}'

        try:
            response = requests.get(search_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [a['href'] for a in soup.find_all('a', href=True)]

            # Filter URLs
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
    st.set_page_config(page_title="Content Optimization for Featured Snippets", layout="wide")

    st.title("Content Optimization for Featured Snippet Success")

    # Initialize NLTK
    setup_nltk()

    # Initialize optimizer
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
                competitor_headings = optimizer.analyze_content_structure(competitor_content)
                competitor_tone = optimizer.analyze_tone(competitor_content)
                competitor_depth = optimizer.analyze_depth(competitor_content)
                competitor_keywords = optimizer.analyze_keywords(competitor_content, [target_keyword])

                st.write(f"**Competitor Headings Structure:** {competitor_headings}")
                st.write(f"**Competitor Tone:** {competitor_tone}")
                st.write(f"**Competitor Content Depth:** {competitor_depth}")
                st.write(f"**Competitor Keyword Relevance:** {competitor_keywords}")

                # Analyze user content
                user_headings = optimizer.analyze_content_structure(user_content)
                user_tone = optimizer.analyze_tone(user_content)
                user_depth = optimizer.analyze_depth(user_content)
                user_keywords = optimizer.analyze_keywords(user_content, [target_keyword])

                st.write(f"**User Headings Structure:** {user_headings}")
                st.write(f"**User Tone:** {user_tone}")
                st.write(f"**User Content Depth:** {user_depth}")
                st.write(f"**User Keyword Relevance:** {user_keywords}")

                # Get semantic keyword suggestions
                semantic_keywords = optimizer.get_semantic_keywords(user_content)
                if semantic_keywords:
                    st.subheader("Semantic Keyword Suggestions")
                    for keyword in semantic_keywords:
                        st.markdown(f"- {keyword}")

                # Get competitor URLs from Google SERP
                if target_keyword.strip():
                    competitor_urls = optimizer.get_competitor_urls_from_serp(target_keyword)

                    if competitor_urls:
                        st.subheader("Suggested Competitor URLs for Further Analysis")
                        for url in competitor_urls:
                            st.write(url)

                # Actionable recommendations
                st.subheader("Actionable Recommendations")
                st.write("1. Improve heading structure with clear H2 and H3 tags.")
                st.write("2. Enhance keyword optimization, including related keywords and semantic terms.")
                st.write("3. Add detailed examples or case studies to improve content depth.")
                st.write("4. Update content with the latest data and industry trends.")
                st.write("5. Focus on concise, engaging language for better readability.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please check your input and try again.")

if __name__ == "__main__":
    main()
