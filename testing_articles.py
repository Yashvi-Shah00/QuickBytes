import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import pandas as pd
import networkx as nx
import rouge
import sys
sys.path.insert(1, 'D:\School\Sem 7\Major project\Classification\Preprocessing')
from normalization import normalize_corpus

# def read_article(file_name):
    # file = open(file_name, "r")
    # filedata = file.readlines()
def read_article(filedata):
    print("Article:",filedata)
    # input_article = request.form["z1"]             

    # print("Article:",input_article)
    # article=[]
    article = list(filedata.split(". "))
    print("article:",article)


    # article=[]
    # for data in filedata:
    #   single_article = data.split(". ")
    #   # article.append(single_article)
    #   article = np.concatenate((article, single_article))
    # # print("article",article)

    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("^[a-zA-Z0-9!@#$&()-`+,/\"]", " ").split(" "))
    sentences.pop() 
    
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

#df = pd.read_csv('D:\School\Sem 7\Major project\Classification\dataset\DemoDataset.csv')
df = pd.read_csv('D:\School\Sem 7\Major project\Classification\dataset\DemoDataset.csv')
df.head()

#Taking a subset of the data:
#df=df.iloc[1:10]

df.info()

df.head()

def generate_summary(data):
    top_n =10
    print('data',data)
    text =data
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(text)
    if not len(sentences)>0:
        summarized_article = list(text.split(". "))
        return summarized_article
    print("sentences:",sentences)
    normalized_sentences = normalize_corpus(sentences)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(normalized_sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank_numpy(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      if(i<len(ranked_sentence)):
       print("print ranked_sentence1",ranked_sentence[i])
       print("print ranked_sentence",ranked_sentence[i][1]) 
       summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))
    summarizedTextOutput =  ". ".join(summarize_text)

    
    return summarizedTextOutput
   # f = open("/content/OutputSummary.txt", "w")
    #f.write(summarizedTextOutput)
    #f.close()

# generate_summary( "Article.txt", 7)
df['Generated_Summary']=  df['CONTENT'].map(generate_summary)
df.dropna(subset = ['Generated_Summary'],inplace=True)

df.head(10)

print("System Generated Summary:\n")
print(df.iloc[3]['GOLDEN_SUMMARY'])
print("\nSummary\n")
print(df.iloc[3]['Generated_Summary'])

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

total_p =0
total_r =0
total_f1 =0
count =0

total_f1 =0
def calculate_rouge_score(rowdata):
  print("len(rowdata)",len(rowdata))
  print("rowdata[2]",rowdata[2])
  print("rowdata[3]",rowdata[3])
  #print('rowdata[4]',rowdata[4])
  output = rowdata[2]
  new_output = re.sub('[^a-zA-Z0-9 \n\.]', '', str(output))
  reference = rowdata[3]
  new_reference = re.sub('[^a-zA-Z0-9 \n\.]', '', str(reference))
  rouge_score = scorer.score(new_output,new_reference)
  # print("rouge_score",rouge_score)
  # total_p = total_p + scores1['rougeL'][0]
  # total_r = total_r + scores1['rougeL'][1]
  total_f1 =  rouge_score['rougeL'][2]
  # print("total_f1",total_f1)
  #count = count + 1
  # return total_p, total_r, total_f1
  return total_f1

# df['Rouge']=  df.apply(calculate_rouge_score, axis=1)
# df.dropna(subset = ['Rouge'],inplace=True)

# df.head(10)

# rouge_result = df["Rouge"].mean()
# print('rouge_score=',rouge_result)
# rouge_result_list= [rouge_result]
# df['Result'] = rouge_result_list
#df['result'] = df["Rouge"].mean() 
df.to_csv('demodatsetresult.csv')
#df['Generated_Summary']=  df['CONTENT'].map(generate_summary)
#df.dropna(subset = ['Generated_Summary'],inplace=True)