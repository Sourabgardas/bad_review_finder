import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('omw-1.4')



content = st.container()

with content:
    header = st.title('Bad review finder')
    file = st.file_uploader("upload the file to be checked")
    if file is not None:
        df = pd.read_csv(file)
    df = df[['Text', 'Star']]


    def clean(text):
        text = re.sub('[^A-Za-z]+', ' ', text)
        return text


    df['Text'] = df['Text'].astype(str)
    df['Cleaned'] = df['Text'].apply(clean)
    pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}


    def token_stop_pos(text):
        tags = pos_tag(word_tokenize(text))
        newlist = []
        for word, tag in tags:
            if word.lower() not in set(stopwords.words('english')):
                newlist.append(tuple([word, pos_dict.get(tag[0])]))
        return newlist


    df['POS tagged'] = df['Cleaned'].apply(token_stop_pos)
    wordnet_lemmatizer = WordNetLemmatizer()


    def lemmatize(pos_data):
        lemma_rew = " "
        for word, pos in pos_data:
            if not pos:
                lemma = word
                lemma_rew = lemma_rew + " " + lemma
            else:
                lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
                lemma_rew = lemma_rew + " " + lemma
        return lemma_rew


    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    analyzer = SentimentIntensityAnalyzer()


    def vadersentimentanalysis(review):
        vs = analyzer.polarity_scores(review)
        return vs['compound']


    df['Vader Sentiment'] = df['Lemma'].apply(vadersentimentanalysis)


    def vader_analysis(compound):
        if compound >= 0.5:
            return 'Positive'
        elif compound <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'


    df['Vader Analysis'] = df['Vader Sentiment'].apply(vader_analysis)
    A = df[df['Vader Analysis'] == 'Positive']
    B = A[df['Star'] <= 2]
    B = B[['Text', 'Star']]
    B
