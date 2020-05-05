# -*- coding: utf-8 -*-

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
from nltk import FreqDist
nltk.download('stopwords') #run this one time

import pandas as pd
pd.set_option("display.max_colwidth", 300) #to set width to show text

import numpy as np
import regex
import spacy

import matplotlib.pyplot as plt
import seaborn as sns


# import data and remove ... and hotel responses
df = pd.read_csv('ReviewData.csv')
df.head(10)
df.shape

df = df[~df['Review'].str.endswith('...')][~df['Review'].str.contains('Thank you|Dear|We are so|We are pleased|We are thankful|Thank|thank|Comment|comment|glad|Glad|Review|review')]
df.shape


#define function to plot most frequent term

def Most_freq_words(x, terms = 30):
	all_words = ' '.join([text for text in x])
	all_words = all_words.split()

	
	fdist = FreqDist(all_words)
	words_df = pd.DataFrame({'word':list(fdist.keys()),
				 'count':list(fdist.values())})
	
	#visualize top most frequent words
	d = words_df.nlargest(columns='count', n=terms)
	plt.figure(figsize=(20,5))
	ax = sns.barplot(data=d, x='word', y='count')
	ax.set(ylabel = 'Count')
	plt.show()
    
Most_freq_words(df['Review'])


#remove unwanted characters, numbers and symbols
df['Review'] = df['Review'].str.replace("[^a-zA-Z#]"," ")

#remove stopwords amd short words(<2letteres)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
type(stop_words)
print(stop_words[1:5])

unwanted_word = ['hotel','staff','hotels','Double','double','tree',
                 'Wyndham','Hyatt','ritz','Suite','suite','par']
for word in unwanted_word:
    stop_words.append(word)

#define function to remove stopwords
def remove_stopwords_unwant(rev):
	rev_new = " ".join([i for i in rev if i not in stop_words])
	return rev_new

#remove short words (length<3)
df['Review'] = df['Review'].apply(lambda x: ' '.join([w for w in x.split() 
                                                    if len(w)>2]))
df['Review']


#remove stopwords from the text and put it to new dataframe
reviews = [remove_stopwords_unwant(r.split()) for r in df['Review']]
reviews[1:5]
type(reviews)

#make entire text lowercase
reviews = [r.lower() for r in reviews]

Most_freq_words(reviews)


#install this python -m spacy download en in shell first
nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization_get_N_Adj(texts, tags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output


#tokenize the review reviews 
tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
print(tokenized_reviews[0:5])
type(tokenized_reviews)

#lemmatize reviews
reviews_afterlemma = lemmatization_get_N_Adj(tokenized_reviews)
print(reviews_afterlemma[0:5])
type(reviews_afterlemma)

#join text back to look like sentences
review_afterlemma_jointext = []

for i in range(len(reviews_afterlemma)):
   review_afterlemma_jointext.append(' '.join(reviews_afterlemma[i]))
type(review_afterlemma_jointext)
review_afterlemma_jointext

Most_freq_words(review_afterlemma_jointext)
review_afterlemma_jointext[0:3]


#LDA model
n_samples = 4800
n_features = 1000
n_components = 7
n_top_words = 20

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.9, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(review_afterlemma_jointext)
print("done in %0.3fs." % (time() - t0))

tf
print(tf_vectorizer.transform(review_afterlemma_jointext)[0:3])

#Fitting LDA
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))


print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


#Print the words and Frequency of each word in topic 

topic_word_distributions = np.array([row / np.sum(row)
                                     for row in lda.components_])
    
np.sum(topic_word_distributions[0])

print('Displaying the top', n_top_words, 'words per topic and their probabilities within the topic...')
print()


for topic_idx in range(n_components):
    print('[ Topic', topic_idx, ']')
    sort_indices = np.argsort(topic_word_distributions[topic_idx])[::-1]  # highest prob to lowest prob word indices
    for rank in range(n_top_words):
        word_idx = sort_indices[rank]
        print(tf_vectorizer.get_feature_names()[word_idx], ':', topic_word_distributions[topic_idx, word_idx])
    print()


