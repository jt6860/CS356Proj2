## Credit to Rowhit Swami/Ragnar from Kaggle for foundation of solution.
## John Torres / CS356

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
        
# Constants
PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""" 
NUM_KEYWORDS = 10 # number of keywords to retrieve in a ranked document
    
def clean_text(text):
    # Lowering text
    text = text.lower()
    
    # Removing punctuation
    text = "".join([c for c in text if c not in PUNCTUATION])
    
    # Removing whitespace and newlines
    text = re.sub('\s+',' ',text)
    
    return text

# Sort a dict with highest score
def sort_coo(coo_matrix):
    
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature, score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def get_keywords(vectorizer, feature_names, doc):
    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])
    
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only TOP_K_KEYWORDS
    keywords=extract_topn_from_vector(feature_names,sorted_items,NUM_KEYWORDS)
    
    return list(keywords.keys())


data = pd.read_csv('./tweets.csv')
data.head()

data.dropna(subset=['text'], inplace=True)
data['text'] = data['text'].apply(clean_text)
data.head()

corpora = data['text'].to_list()

# Initializing TF-IDF Vectorizer with stopwords
vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True, use_idf=True)

# Creating vocab with our corpora
# Exclluding first 10 docs for testing purpose
vectorizer.fit_transform(corpora[10::])

# Storing vocab
feature_names = vectorizer.get_feature_names_out()

result = []
for doc in corpora[0:15000]:
    df = {}
    df['text'] = doc
    df['top_keywords'] = get_keywords(vectorizer, feature_names, doc)
    result.append(df)
    
final = pd.DataFrame(result)

data['Top Keywords'] = final['top_keywords']

data.to_csv('Tweets.csv')
data.to_excel('Tweets.xlsx')

print(data)