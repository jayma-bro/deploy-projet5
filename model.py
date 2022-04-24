import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
import numpy as np
import json
import joblib
with open('stop_words_english.json') as json_file:
    stops = json.load(json_file)

class Model:
    def __init__(self) -> None:
        tags_binarizer = joblib.load('tags_binarizer.pkl')
        self.classes = tags_binarizer.classes_
        self.tf_idf = joblib.load('tf_idf.pkl')
        self.count_vect = joblib.load('count_vect.pkl')
        self.model = joblib.load('final_model.pkl')
        self.lda = joblib.load('lda.pkl')
        self.ldaTags = np.vectorize(lambda x: self.count_vect.get_feature_names_out()[x])(self.lda.components_.argmax(axis=1))
    def predict(self, title: str, body: str) -> dict:
        words: str = self.post_to_words(body + ' ' + title)
        vect = self.count_vect.transform([words])
        tf_idf = self.tf_idf.transform(vect)
        tag_proba = np.array(self.model.predict_proba(tf_idf))[0]
        topic_proba = self.lda.transform(vect)[0]
        best_tag = []
        best_topic = []
        for i in range(6):
            best_tag.append(self.classes[tag_proba.argmax()])
            tag_proba[tag_proba.argmax()] = 0
            best_topic.append(self.ldaTags[topic_proba.argmax()])
            topic_proba[topic_proba.argmax()] = 0
        return {'tags': best_tag, 'topic': best_topic}
    
    def post_to_words(self, post_text: str, stopwords_lem=True) -> str:
        # 2. Remove non-letters
        post_text = re.sub(r"(\\n)|(\.\\n)|(\'\w+)|(http*\S+)|[^\w\s\.,#+-]", " ", post_text)
        post_text = re.sub(r"(\.\s)|(\.$)|(,\s)|(\s#\s)|(\'\w+)|(\s\-\s)", " ", post_text)
        post_text = re.sub(r"(\s-?\+?\d{0,4}\s)", " ", post_text)
        post_text = re.sub(r"\s+", " ", post_text)
        #
        # 3. Convert to lower case, split into individual words
        words = post_text.lower().split()
        #
        if stopwords_lem :
            # 4. In Python, searching a set is much faster than searching
            # a list, so convert the stop words to a set
            # 5. Remove stop words
            words = [w for w in words if not w in stops]
            # 6. Join the words back into one string separated by space,
            # and return the result.
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join( words )
    
    def clean_html_body(self, text: str) -> str:
        soup = BeautifulSoup(text, features="lxml")
        code_to_remove = soup.findAll("code")
        for code in code_to_remove:
            code.replace_with(" ")
        return soup.get_text()