# -*- coding: utf-8 -*-
"""
Descrption: Feature Mining is meant to understand industry discussion points 
            on corporate performance from sources such as earning calls and financial reports.
            
"""
#pip install nameparser
#pip install textblob
#pip install sumy

from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.tokenize import sent_tokenize,word_tokenize
import nltk
from nltk.tag.stanford import CoreNLPNERTagger
from nltk.tag.stanford import StanfordNERTagger 

import os
from nameparser.parser import HumanName
import numpy as np
import pandas as pd
from nltk.corpus import stopwords,wordnet
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob, Word
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from collections import Counter 

from nltk import sent_tokenize
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

def Get_Speech(Audio_Text):
    start_tag=Audio_Text.find('Operator')
    end_tag=Audio_Text.find('Q & A')
    Audio_Text=Audio_Text[(start_tag+11):(end_tag-1)]
    return Audio_Text

def Get_QA(Audio_Text):
    start_tag=Audio_Text.find('Q & A')
    end_tag=Audio_Text.find('mainEntityOfPage')
    Audio_Text=Audio_Text[(start_tag+42):(end_tag-3)]
    return Audio_Text


def GetSummary(Filename,SummaryType,Section='All'):   
    """
        GetSummary() is used to determine the summary by product, opinions and general discussion points.

        Parameters
        ----------
        Filename : path to the file to assess.
        SummaryType: The summary type aim to identify products(NNP), opinions(JJ) and general discussion points(NN).

        Returns
        -------
        Description.
    """  
    # #read File 
    file = open(Filename, 'r')
    Audio_Text=file.read()
    file.close()

    # #confirm section
    if(Section=='Q & A'):
        Audio_Text=Get_QA(Audio_Text)    
    elif(Section=='Exec Speech'):
        Audio_Text=Get_Speech(Audio_Text)

    #***********word tokens***********#
    #word_tokenize based on the text(unit is whole text)
    words = word_tokenize(Audio_Text)
    #function to test if something is a noun
    is_noun = lambda pos: pos[:3] == SummaryType  #extract part between pos1,pos2
    #nouns = pd.Series([word for (word, pos) in nltk.pos_tag(words) if is_noun(pos)]).value_counts()
    #nltk.pos_tag(words)：part-of-speech tagging , or word classes
    nouns = pd.Series([word for (word, pos) in nltk.pos_tag(words) if is_noun(pos)])
    # #clean  
    temp=pd.Series(nouns).str.replace('\W', ' ')
    temp=temp.replace(' ', np.nan).dropna()  
    temp = " ".join(temp) #connect nouns
    #tokenize nouns
    Nouns_Cleaned = word_tokenize(temp)    
    #convert to lower case
    Nouns_Cleaned = [w.lower() for w in Nouns_Cleaned]
    # #get standard stop words. Review final results for other additional stop words.
    stop_words = stopwords.words('english')
    custom_words=['']
    stop_words.extend(custom_words)
    #drop stopwords
    Nouns_Cleaned = [i for i in Nouns_Cleaned if i not in stop_words] 
    #lemmetize words
    nouns_lemmatized = [WordNetLemmatizer().lemmatize(w) for w in Nouns_Cleaned]

    #***********find person names***********#
    #word_tokenize based on the sent_tokenize(unit is each sentence in the text)
    tokenized_sents = [word_tokenize(sent) for sent in sent_tokenize(Audio_Text)]

    # #1. Stanford Method
    #Stanford Method (Note, this needs to be commented and needs toe be converted to an api version and/or investigate cloud version)
    #java_path = C:/Program Files/Java/jdk-14.0.1/bin/java.exe
    #java_path = '/Library/Java/JavaVirtualMachines/jdk-11.0.2.jdk/Contents/Home/bin/java'
    java_path ='/usr/lib/jvm/java-11-openjdk-amd64'
    os.environ['JAVAHOME'] = java_path
    #nltk.internals.config_java('C:/Program Files/Java/jdk-14.0.1/bin/java.exe')
    #nltk.internals.config_java("/Library/Java/JavaVirtualMachines/jdk-11.0.2.jdk/Contents/Home/bin/java")
    nltk.internals.config_java('/usr/lib/jvm/java-11-openjdk-amd64/bin/java')
    #st = StanfordNERTagger(stanford-ner/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz,stanford-ner/stanford-ner-2017-06-09//stanford-ner.jar)
    st=StanfordNERTagger('/content/drive/MyDrive/Earnings_call_NLP/stanford-ner/english.all.3class.distsim.crf.ser.gz',
                         '/content/drive/MyDrive/Earnings_call_NLP/stanford-ner/stanford-ner-2017-06-09/stanford-ner.jar')
    tags=st.tag_sents(tokenized_sents)   #('With', 'O'),('that', 'O'),('Mike', 'PERSON'),
    #get person names list 
    names_stanford=[]
    for tag in tags:
        for content in tag:
            if content[1]=='PERSON':
                names_stanford.extend(content)  #'Mike','PERSON', 'Spencer', 'PERSON',
    #keep names and remove 'PERSON' tag
    names_stanford=[i for i in names_stanford if i not in ['PERSON']]

    # #2.WordNet Method
    person_list=[]
    names_wordnet=person_list
    def get_human_names(text):
        tokens = nltk.tokenize.word_tokenize(text)
        #tag
        pos = nltk.pos_tag(tokens)
        sentt = nltk.ne_chunk(pos, binary = False)
        person = []
        name = ""
        for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
            for leaf in subtree.leaves():
                person.append(leaf[0])
            if len(person) > 1: #avoid grabbing lone surnames
                for part in person:
                    name += part + ' '
                if name[:-1] not in person_list:
                    person_list.append(name[:-1])
                name = ''
            person = []
    #     print (person_list)   
    names = get_human_names(Audio_Text)
    for person in person_list:
        person_split = person.split(" ")
        for name in person_split:
            if wordnet.synsets(name):
                if(name in person):
                    names_wordnet.remove(person)
                    break
    names_wordnet = [word_tokenize(w) for w in names_wordnet]
    names_wordnet = [item for sublist in names_wordnet for item in sublist]

    #***********remove names***********#
    #Remove Duplicates 
    names_stanford = list(dict.fromkeys(names_stanford))
    names_wordnet= list(dict.fromkeys(names_wordnet))
    #Convert to Lower
    names_stanford = [w.lower() for w in names_stanford]
    names_wordnet = [w.lower() for w in names_wordnet]  
    #same names in 2 methods
    common_names=[i for i in names_stanford if i in names_wordnet] 
    #Remove names from tokens
    Nouns_Cleaned = [i for i in Nouns_Cleaned if i not in common_names] 
    #Clean to remove any orphaned works split as part of special character removals
    Nouns_Cleaned = [i for i in Nouns_Cleaned if len(i)>2]
    #count frequency
    Description=pd.Series(Nouns_Cleaned).value_counts()
    return Description



def ResultSummary(Nouns_Cleaned,Description):
    """
        ResultSummary is used to report the output from the GetSummary function.

        Parameters
        ----------
        Nouns_Cleaned : A list of all words in a particular category.
        Description: The summary of the Noun_Cleaned file.

        Returns
        -------
        Nouns_Cleaned,Description.
    """
    # Plot a bar graph on top buzz words
    Top_Percentile= Description.sort_values(0,ascending=False)[0:30]
    fig = plt.figure(figsize=(10, 10))
    plt.bar(Top_Percentile.index,Top_Percentile.values)
    plt.xticks(rotation=45)
    plt.title('Buzz Words')
    plt.xlabel('Words')
    plt.ylabel('Occurence Count')
    plt.show()  
    #WordCount on all discussions
    from wordcloud import WordCloud,STOPWORDS
    from  functools import reduce
    wordcloud = WordCloud(background_color='white',
                          width=2500,
                          height=2000
                         ).generate_from_frequencies(frequencies=dict(Nouns_Cleaned)) 
    plt.figure(1,figsize=(10, 10))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis('off')
    plt.show()
    
    
def CombineResults(df_Combined,Description):
    """
        CombineResults is used to combine results from multiple reports.

        Parameters
        ----------
        df_Combined : A combined list of all features and frequency.
        Description: The summary of the features and frequency in current file.

        Returns
        -------
        new_dict: A dictionary that can be used in a tagcloud for output visualization.
        df_Combined.
    """   
    df_Des_1=pd.DataFrame(Description,columns={"Counts"})
    df_Combined=pd.concat([df_Combined,df_Des_1])
    df_Combined=df_Combined.groupby(level=0).sum()
    All_Details=df_Combined.iloc[:,0]
    df_dict=df_Combined.to_dict('dict')
    for key, value in df_dict.items(): 
        new_dict=value
    
    return df_Combined,All_Details,new_dict


def Calculate_Accuracy(Scores):
    Scores_RMSE=pd.DataFrame(columns=["PySentiment","TextBlob","Vader","FinBert"])
    Score_Accuracy=pd.DataFrame(columns=["PySentiment","TextBlob","Vader","FinBert"])
    Scores_RMSE=Scores_RMSE.append({"PySentiment":mean_squared_error(Scores["Actual_Score"], Scores["PySentiment_Score"],squared = False),
                                    "TextBlob":mean_squared_error(Scores["Actual_Score"], Scores["TextBlob_Score"],squared = False),
                                    "Vader":mean_squared_error(Scores["Actual_Score"], Scores["Vader_Score"],squared = False),
                                    "FinBert":mean_squared_error(Scores["Actual_Score"], Scores["FinBert_Score"],squared = False)
                                    },ignore_index = True)
    
    Count_Finbert=Scores.groupby(["FinBert_Sentiment", "Actual_Sentiment"]).size().reset_index(name="Count")
    Count_Vader=Scores.groupby(["Vader_Sentiment", "Actual_Sentiment"]).size().reset_index(name="Count")
    Count_TextBlob=Scores.groupby(["TextBlob_Sentiment", "Actual_Sentiment"]).size().reset_index(name="Count")
    Count_PySentiment=Scores.groupby(["PySentiment_Sentiment", "Actual_Sentiment"]).size().reset_index(name="Count") 
    Score_Accuracy=Score_Accuracy.append({"PySentiment":(Count_PySentiment.loc[Count_PySentiment['PySentiment_Sentiment']==Count_PySentiment['Actual_Sentiment'],['Count']].sum(axis=0)[0])/Count_PySentiment.sum(axis=0)['Count'],
                                          "TextBlob":(Count_TextBlob.loc[Count_TextBlob['TextBlob_Sentiment']==Count_TextBlob['Actual_Sentiment'],['Count']].sum(axis=0)[0])/Count_TextBlob.sum(axis=0)['Count'],
                                          "Vader":(Count_Vader.loc[Count_Vader['Vader_Sentiment']==Count_Vader['Actual_Sentiment'],['Count']].sum(axis=0)[0])/Count_Vader.sum(axis=0)['Count'],
                                          "FinBert":(Count_Finbert.loc[Count_Finbert['FinBert_Sentiment']==Count_Finbert['Actual_Sentiment'],['Count']].sum(axis=0)[0])/Count_Finbert.sum(axis=0)['Count']
                                          },ignore_index = True)
    return Scores_RMSE,Score_Accuracy


def Get_SentimentProgress(Sentence_Scores,Stock_Ticker,Filename):  
    qtr=Filename[len(Filename)-11:len(Filename)-4]
    Sentiement_Progress=pd.DataFrame(columns=["PySentiment_Sentiment","TextBlob_Sentiment","Vader_Sentiment","FinBert_Sentiment"])

    Sentences=Sentence_Scores[Stock_Ticker][qtr]["Sentence_Score"]["FinBert"][0]['sentence']
    Sentence_Base=Sentence_Scores[Stock_Ticker][qtr]["Sentence_Score"]
    counter=0
    for sentence in Sentences:
        Sentiement_Progress=Sentiement_Progress.append({"PySentiment_Sentiment":0,
                                                        "TextBlob_Sentiment":0,
                                                        "Vader_Sentiment":0,
                                                        "FinBert_Sentiment":0
                                                        },ignore_index = True)
        #1
        temp=Sentence_Base['PySentiment'][counter]['Polarity']
        Sentiement_Progress["PySentiment_Sentiment"][counter]=-1 if float(temp)<-0.05 else  (1 if float(temp)>0.05 else 0)
        #2
        temp=Sentence_Base['TextBlob'][counter][0]
        Sentiement_Progress["TextBlob_Sentiment"][counter]=-1 if float(temp)<-0.05 else  (1 if float(temp)>0.05 else 0)
        #3
        temp=Sentence_Base['Vader'][counter]['compound']
        Sentiement_Progress["Vader_Sentiment"][counter]=-1 if float(temp)<-0.05 else  (1 if float(temp)>0.05 else 0)
        #4
        temp=Sentence_Base['FinBert'][0]['prediction'][counter]
        Sentiement_Progress["FinBert_Sentiment"][counter]=-1 if str(temp)=="negative" else  (1 if str(temp)=="positive" else 0)
        
        counter+=1
    
    Sentiement_Progress=Sentiement_Progress.cumsum(axis=0) #Accumulate by row
    ax = Sentiement_Progress.plot()
    ax.set_xlabel("Meeting Progress", labelpad=5, weight='bold', size=8)
    ax.set_ylabel("Sentiment Level", labelpad=5, weight='bold', size=8)
    ax.set_title('Comparison of Sentiment Scores',weight='bold', size=12)





#pip install -U gensim ##Gensim
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

def Summarize_Sent_Count(Filename,Section,Summarize_Method):    
    file = open(Filename, 'r')
    Audio_Text=file.read()
    file.close()
    
    if(Section=='Q & A'):
        Audio_Text=Get_QA(Audio_Text)    
    elif(Section=='Exec Speech'):
        Audio_Text=Get_Speech(Audio_Text)
    
    sentences_count=int(len(sent_tokenize(Audio_Text))*0.05)  #top 5% which is 8 sentences
    return sentences_count 
   


def Summarize_Content_Custom(Audio_Text,sentences_count,Summarize_Method):     
    actual_sentences_count=float(len(sent_tokenize(Audio_Text)))*0.5
    parser = PlaintextParser.from_string(Audio_Text,Tokenizer("english"))
    stemmer = Stemmer("english")
    if(Summarize_Method=="Gensim"):
        #ratio: define length of the summary as a proportion of the text 
        temp=summarize(Audio_Text,ratio=0.5)
        sen=sent_tokenize(temp)
        sen=Counter(sen)
        temp=sen.most_common(sentences_count)
        for value in temp:
            print(value[0])
    elif(Summarize_Method=="LexRankSummarizer"):        
        # Using LexRank(Sentence based ranking based on repeating sentences)        
        summarizer_Lex = LexRankSummarizer(stemmer)    
        summarizer_Lex.stop_words = get_stop_words("english")
        #Summarize the document with 2 sentences
        summary = summarizer_Lex(parser.document, actual_sentences_count)
        sen=Counter(summary)
        temp=sen.most_common(sentences_count)
        for value in temp:
            print(value[0])
    elif(Summarize_Method=="LuhnSummarizer"):
         # Using LUHN(Sentence based on frequency of most important words)
        summarizer_luhn = LuhnSummarizer(stemmer)
        summarizer_luhn.stop_words = get_stop_words("english")
        summary_1 =summarizer_luhn(parser.document,actual_sentences_count)
        sen=Counter(summary_1)
        temp=sen.most_common(sentences_count)
        for value in temp:
            print(value[0])
    elif(Summarize_Method=="LsaSummarizer"):
        # Using LSA(Sentence based on frequency of most important words)
        summarizer_lsa2 = LsaSummarizer()
        summarizer_lsa2 = LsaSummarizer(stemmer)
        summarizer_lsa2.stop_words = get_stop_words("english")   
        summary = summarizer_lsa2(parser.document,actual_sentences_count)
        sen=Counter(summary)
        temp=sen.most_common(sentences_count)
        for value in temp:
            print(value[0])
    elif(Summarize_Method=="TextRankSummarizer"):
        # Using LSA(Sentence based on frequency of most important words)
        summarizer_text = TextRankSummarizer()
        summarizer_text = TextRankSummarizer(stemmer)
        summarizer_text.stop_words = get_stop_words("english")    
        summary = summarizer_text(parser.document,actual_sentences_count)
        sen=Counter(summary)
        temp=sen.most_common(sentences_count)
        for value in temp:
            print(value[0])
'''  
 def Summarize_Content(Filename,Section,Summarize_Method):    
    #Read File 
    file = open(Filename, 'r')
    Audio_Text=file.read()
    file.close()
    
    if(Section=='Q & A'):
        Audio_Text=Get_QA(Audio_Text)    
    elif(Section=='Exec Speech'):
        Audio_Text=Get_Speech(Audio_Text)
    
    sentences_count=int(len(sent_tokenize(Audio_Text))*0.05)  #top 5% which is 8 sentences
    #
    parser = PlaintextParser.from_string( Audio_Text,Tokenizer("english") )
    stemmer = Stemmer("english")  
    if(Summarize_Method=="Gensim"):
        print(summarize(Audio_Text,ratio=0.05))      
    elif(Summarize_Method=="LexRankSummarizer"):        
        # Using LexRank(Sentence based ranking based on repeating sentences)        
        summarizer_Lex = LexRankSummarizer(stemmer)    
        summarizer_Lex.stop_words = get_stop_words("english")
        #Summarize the document with 8 sentences
        summary = summarizer_Lex(parser.document, sentences_count)
        for sentence in summary:
            print(sentence)
    elif(Summarize_Method=="LuhnSummarizer"):
         # Using LUHN(Sentence based on frequency of most important words)
        summarizer_luhn = LuhnSummarizer(stemmer)
        summarizer_luhn.stop_words = get_stop_words("english")
        summary_1 =summarizer_luhn(parser.document,sentences_count)
        for sentence in summary_1:
            print(sentence)
    elif(Summarize_Method=="LsaSummarizer"):
        # Using LSA(Sentence based on frequency of most important words)
        summarizer_lsa2 = LsaSummarizer()
        summarizer_lsa2 = LsaSummarizer(stemmer)
        summarizer_lsa2.stop_words = get_stop_words("english")    
        for sentence in summarizer_lsa2(parser.document,sentences_count):
            print(sentence)          
    elif(Summarize_Method=="TextRankSummarizer"):
        # Using LSA(Sentence based on frequency of most important words)
        summarizer_text = TextRankSummarizer()
        summarizer_text = TextRankSummarizer(stemmer)
        summarizer_text.stop_words = get_stop_words("english")    
        for sentence in summarizer_text(parser.document,sentences_count):
            print(sentence)
    #   
    return sentences_count  
'''     