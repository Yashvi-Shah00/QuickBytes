#importing the neccessary libraries

import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
import string
from nltk.corpus import wordnet 
#from pattern3.en import tag
from contractions import CONTRACTION_MAP

lemmatizer = WordNetLemmatizer()
#lemmatizer = nltk.stem.WordNetLemmatizer()
    
#Function for expanding the contraction. Eg n't = not

def expand_contractions(text):
    
    contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTION_MAP.get(match)\
                                if CONTRACTION_MAP.get(match)\
                                else CONTRACTION_MAP.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

#Function for removing the stopwords

stopword_list = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    filtered_words = [word for word in nltk.word_tokenize(text) if word not in stopword_list and word.isalpha()]
    filtered_text = ' '.join(filtered_words)    
    return filtered_text


#Function for removing special characters
def remove_special_characters(text):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', word) for word in nltk.word_tokenize(text)])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

#lemmatization
#def pos_tag_text(text):
    
    #def penn_to_wn_tags(pos_tag):
    #     if pos_tag.startswith('J'):
    #         return wn.ADJ
    #     elif pos_tag.startswith('V'):
    #         return wn.VERB
    #     elif pos_tag.startswith('N'):
    #         return wn.NOUN
    #     elif pos_tag.startswith('R'):
    #         return wn.ADV
    #     else:
    #         return None
    
    # tagged_text = tag(text)
    # tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
    #                      for word, pos_tag in
    #                      tagged_text]
    # return tagged_lower_text
    
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
# lemmatize text based on POS tags    
# def lemmatize_text(text):
#     print('lemmatize_text',text)
#     pos_tagged_text = pos_tag_text(text)
#     lemmatized_tokens = [lemmatizer.lemmatize(word, pos_tag) if pos_tag
#                          else word                     
#                          for word, pos_tag in pos_tagged_text]
#     lemmatized_text = ' '.join(lemmatized_tokens)
#     return lemmatized_text

def lemmatize_text(text):
    print('lemmatize_text',text)
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:	
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)

    #print(lemmatized_sentence)
    return lemmatized_sentence

def normalize_corpus(text):
    print('isinstance(text,list)',isinstance(text,list))
    normalized_text =[]
    if(isinstance(text,list)):
        for single_sentence in text:
            sentence = str(single_sentence)
            sentence = expand_contractions(sentence)
            sentence = remove_stopwords(sentence)
            sentence = lemmatize_text(sentence)
            sentence = remove_special_characters(sentence)
            normalized_text.append(sentence)
        return normalized_text
    else:        
        text1 = str(text)
        text1 = expand_contractions(text1)
        text1 = remove_stopwords(text1)
        text1 = lemmatize_text(text1)
        text1 = remove_special_characters(text1)
        return text1


def parse_content(text):
    document=[]
    for paragraph in text.split('\n'):
        if paragraph!="":
            for sent in nltk.sent_tokenize(paragraph):
                document.append(sent)
    return document