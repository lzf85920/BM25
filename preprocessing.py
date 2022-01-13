import xml.etree.ElementTree as ET
import pandas as pd
import os
import re
import argparse
import numpy as np
from nltk.corpus import stopwords
import json
import nltk, string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
port_stem = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def extract_query(filepath):
    with open(filepath, 'r') as f:
        content = ''.join(f.readlines())
        content = BeautifulSoup(content, "html.parser")
    return content.summary.text

def extract_body(filepath):
  with open(filepath, 'r') as f:
      content = ''.join(f.readlines())
      content = BeautifulSoup(content, "html.parser")
      content = str(content.body)
  return content

title_list = ['Methods','Measurements','Outcomes','Predictor', 'Study Design', 'Setting &amp; Participants','Limitations', 'Material/Methods', 'Conclusion:', 'Results:', 'Methods:', 'Introduction:', 'Materials and Methods:','Methods and Results', 'Electronic supplementary material', 'Objectives', 'Results', 'Introduction', 'Trial registration', 'Discussion', 'Case presentation', 'Case Report', 'Conclusions', 'Background', 'Aims']

def Lower_casing(text):
    return text.lower()

def Remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def Lemmatization_stemming_stopword(text_list):
    text_list = text_list.split()
    text = [port_stem.stem(wordnet_lemmatizer.lemmatize(t)) for t in text_list if t not in stopwords.words('english')+title_list]

    return ' '.join(text)

def scrub_words(text):
    """Basic cleaning of texts."""
    # remove html markup
    text=re.sub("(<.*?>)"," ",text)
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    return text

def Preprocessing(text):
    text = text.lower()
    text = scrub_words(text)
    text = Remove_punctuation(text)
    text = Lemmatization_stemming_stopword(text)
    return text

title_list = Preprocessing(' '.join(title_list)).split()


