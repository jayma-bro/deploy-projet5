from ray import serve

import matplotlib.pyplot as plt
plt.style.use("seaborn")
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

import json
with open('stop_words_english.json') as json_file:
    stops = json.load(json_file)

serve.start(http_options={'port': 8020}, detached=True)

@serve.deployment(route_prefix="/predict")  # type: ignore

class ModelPipeline:
    def __init__(self):
        print('initialisé')
        
    async def __call__(self, starlette_request):
        request = await starlette_request.json()
        words: str = self.post_to_words(self.clean_html_body(request['body']) + ' ' + request['title'])
        return {"result": words}
    
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
        soup = BeautifulSoup(text)
        code_to_remove = soup.findAll("code")
        for code in code_to_remove:
            code.replace_with(" ")
        return soup.get_text()
ModelPipeline.deploy()  # type: ignore