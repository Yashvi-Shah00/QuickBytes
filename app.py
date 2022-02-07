import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
# from string import punctuation
# from heapq import nlargest
import heapq
from flask import Flask , render_template , request


# todo:Add imports
import nltk
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# app = Flask("Text_summarizer")
app = Flask(__name__)


@app.route("/home")
def home():
        return render_template( "login page.html" )




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
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
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
    top_n = 5
    print("check1")
    # nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    get_sentences =  read_article()
    print("get_sentences:",get_sentences)
    print("stop_words:",stop_words)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(get_sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    print("scores",scores)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(get_sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))
    summarizedTextOutput =  ". ".join(summarize_text)

    # return summary
    print("summary",type(summarizedTextOutput))
    print("summary",summarizedTextOutput)
    return render_template('summarydisplay.html', summary=summarizedTextOutput)

        # print(summarise(text))

app.run(host="localhost" , port=8080)