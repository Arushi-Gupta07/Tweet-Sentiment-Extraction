![giphy (1)](https://user-images.githubusercontent.com/57909909/168902321-760c3caf-864f-4f2c-b5ee-6199863df92f.gif)

# Tweet-Sentiment-Extraction


Problem Definition
In this Kaggle challenge, a tweet and its corresponding sentiment are given. The task is to extract a part of the tweet that leads to the given sentiment.This problem,in common, can be posted as a Question Answering task wherein the sentiment acts as question, tweet as context from which answer is to be extracted and selected_text as the answer.

Dataset
The data set provided by Kaggle had two files: train and test. The train file had 27,481 different tweets, one among the three sentiments (positive, negative or neutral) and their corresponding extracted text (selected_text). The challenge was to predict the selected_text for the 3534 tweets present in test data set.

Performance Metric
The performance metric used for evaluation is word-level Jaccard score. It is calculated as follows:

<img width="714" alt="image" src="https://user-images.githubusercontent.com/116758652/213876294-1924227a-84d7-41cf-9ae7-29e7e0950a6e.png">


Exploratory Data Analysis

1) We have three different types of sentiment. Neutral sentiment has the largest number of tweets followed by positive and negative sentiment.

2) Majority of neutral tweets has jaccard similarity score as 1. This conclude that text and selected text are mostly the same for neutral tweets.

3) There are large number of tweets with word length 5 to 7 irrespective of sentiment value. The tweets which are having number of words greater than 25 are very less and hence, words distribution plot is right skewed.

4) Positive and negative tweets jaccard score have high kurtosis and thus values are concentrated in two regions narrow and high density.

5) Neutral sentiment has higher no. of words in the selected_text compared to postive and negative sentiment. Also, around 92% of neutral tweets have equal word length with selected_text and high Jaccard score of 1. Thus, neutral tweets can be returned as it is as the selected_text.

6) Selected text is copy paste of a part of the text, so cleaning the data may not be feasible here, we will see whether cleaning data improves model performance during modelling.

7) Postive and negative sentiment have similar distribution for difference in word length and jaccard score.

8) We can see from the jaccard score plot that there is peak for negative and positive plot around score of 1. This means there is a cluster of tweets where there is a high similarity between text and selected texts, if we can find those clusters then we can predict text for selected texts for those tweets irrespective of the sentiment type.

9) We created word cloud and got significant textual points which are highlighted with larger size in word cloud and are contibuted as per given sentiment. These plays a key role i the predictions of selected text. From word cloud, we can see words like I, to, the, go, dont, got, u, cant,lol, like are common in all three segments. That's interesting because words like don't and can't are more of negative nature and word like lol is kind of positive nature word.

10) After ploting 20 bar graph of common word found these word also highlighted in word cloud. 

Different Models implemented are:

1) LSTM

2) Bi-LSTM

3) RoBERTA, The final pipeline is build using roberta model as this model perfors best among all.

Medium Blog - https://medium.com/@iamarushigupta07/tweet-sentiment-extraction-a3a3d5037792
