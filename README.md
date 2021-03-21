# Earnings_Call_Analyzed_By_NLP

Collect companies' Earnings Call Transcripts by web scraping and then apply NLP methods to their contents to find the best sentiments analyser.

To test this project, we will apply Microsoft Corporation earnings call as an example.

### Data Source and Pacakages

- US Corporate Earnings Call Transcripts can be reached from https://news.alphastreet.com/

- Stanford NER is a Java implementation of a Named Entity Recognizer, the package we used in this project can be download from https://drive.google.com/drive/folders/1mS0y92w65f4R7u6dNP8KFFAtdmNmI-v9?usp=sharing the file named 'stanford-ner'


### Four NLP analysis methods

- FinBERT
- PySentiment
- TextBlob
- Vader

### Results

Finbert enjoys high accuracy and an acceptable error rate at the same time.

<img src="https://github.com/nilijing/Earnings_Call_Analyzed_By_NLP/blob/main/images/accuracy.png" width="500" />

WordCloud plot shows that 'currency', 'growth','revenue','cloud'  are the most common words mentioned in the latest 12 Microsoft earning calls. 

<img src="https://github.com/nilijing/Earnings_Call_Analyzed_By_NLP/blob/main/images/wordscloud.png" width="500" />

### Reference

FinBERT model introduction：https://paperswithcode.com/paper/finbert-a-pretrained-language-model-for
