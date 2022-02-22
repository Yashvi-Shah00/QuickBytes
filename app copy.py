import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
# from string import punctuation
# from heapq import nlargest
import heapq
from flask import Flask , render_template , request
import random


# todo:Add imports
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import pandas as pd
import networkx as nx
#import rouge
import sys
sys.path.insert(1, 'D:\School\Sem 7\Major project\Final Project\QuickBytes\Preprocessing')
from normalization import normalize_corpus
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

# nltk.download('stopwords')

# app = Flask("Text_summarizer")
app = Flask(__name__)


@app.route("/home")
def home():
        return render_template( "login_page.html" )

@app.route("/input")
def input():
        return render_template( "index.html" )

@app.route("/select_newspaper")
def select_newspaper():
        return render_template( "select_newspaper.html" )

def crawl_website(urlList):
  for i in range(0,5):
    num = random.randint(20, 100)
    print(num)
    required_url = urlList[num]
    final_url_list.append(required_url)

@app.route("/process_newspaper" , methods = ["POST"] )
def display_content():
        newspaper_name = request.form["n1"]
        if newspaper_name == 'Ndtv':
            newspaper_link = 'https://www.ndtv.com/'
        elif newspaper_name == 'Indian Express':
            newspaper_link = 'https://indianexpress.com/'
        else:
            newspaper_link = 'https://www.hindustantimes.com/'

        news_paper1 = newspaper.build(newspaper_link,memoize_articles=False)
        urlData1 = []
        final_url_list = []

        for article in news_paper1.articles:
            urlData1.append(article.url)
        crawl_website(urlData1)
        print(final_url_list)
        return render_template( "list_articles.html" )


def read_article():
    # file = open(file_name, "r")
    # filedata = file.readlines()
    input_article = request.form["z1"]             

    # print("Article:",input_article)
    article=[]
    filedata = list(input_article.split(". "))
    print("filedata:",filedata)

    # for data in filedata:
    # #   single_article = data.split(". ")
    #   article = np.concatenate((article, data))

    # # for data in filedata:
    # #   single_article = data.split(". ")
    # #   article = np.concatenate((article, single_article))
    # print("article:",article)

    sentences = []

    # for sentence in article:
    for sentence in filedata:
        sentences.append(sentence.replace("^[a-zA-Z0-9!@#$&()-`+,/\"]", " ").split(" "))
    sentences.pop() 
    print("sentences",sentences)
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix



@app.route("/process" , methods = ["POST"] )

def prediction():
    top_n = 10
    print("check1")
    # nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    get_sentences =  read_article()
    print("get_sentences:",get_sentences)
    print("stop_words:",stop_words)
    normalized_sentences = normalize_corpus(get_sentences)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(normalized_sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    print("scores",scores)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(get_sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
        if(i<len(ranked_sentence)):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))
    summarizedTextOutput =  ". ".join(summarize_text)

    # return summary
    print("summary",type(summarizedTextOutput))
    print("summary",summarizedTextOutput)
    return render_template('summary_display.html', summary=summarizedTextOutput)

        # print(summarise(text))
@app.route("/input2")
def input2():
        return render_template( "index2.html" )

@app.route("/process_category" , methods = ["POST"] )

def predict_category():
    print("check2")

    get_sentences =  read_article()
    normalized_sentences = normalize_corpus(get_sentences)

    Tfidf_from_pickle = load(open('models/tfidf_model.pkl', 'rb'))
    SVM_from_pickle = load(open('models/svm_model.pkl', 'rb'))

    tfid_test_features1 = Tfidf_from_pickle.transform(normalized_sentences)
    category_output = SVM_from_pickle.predict(tfid_test_features1)
    final_category_output = ''
    if category_output == 0:
        final_category_output = 'business'
    else if category_output == 1: 
        final_category_output = 'entertainment'
    else if category_output == 2:
        final_category_output = 'health'
    else:    
        final_category_output = 'technology'

    # final_category_output = category_output == 0 if 'business' : category_output == 1 ? 'entertainment' : category_output == 2 ? 'health' : 'technology'

    return render_template('summary_display.html', summary=final_category_output)


app.run(host="localhost" , port=8080)