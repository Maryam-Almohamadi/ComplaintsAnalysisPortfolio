import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Downloading esssential NLTK resources for preprocessing
nltk.download('stopwords') # list of common words
nltk.download('punkt') # split text for tokenization, handling punctuation
nltk.download('wordnet') # lexical data base for lemmatization, reduces words to base form

stop_w = set(stopwords.words('english'))
stop_w.add("xxxx") # added "xxxx" to stopwords since it's the redacted credit card number of customer and is not a word

lemmatizer = WordNetLemmatizer()

def cleaning(text):
    text = re.sub(r"\W", " ", text) # converts every non-word symbol in the text to a space
    tokens = word_tokenize(text.lower()) # converts text to lowercase before it separates it into words (tokens)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_w and len(word) > 2]
    # eliminates  stopwords before converting them into their base form using lemmatizer.lemmatize(word)
    return " ".join(tokens)
    # tokens are joined back and returned as a string.