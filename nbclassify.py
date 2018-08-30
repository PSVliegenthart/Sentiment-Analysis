"""
    Logic for machine learning
"""
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
import codecs
import pandas as pd
import re
import nltk
from collections import Counter
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from time import time
import argparse
import numpy as np

#Set the arg parser
parser = argparse.ArgumentParser(description='Train a classifier and predict text emotion')
parser.add_argument('-T'    ,'--train'      ,default = 'train.csv'      ,   help='The name of the training dataset file. (default : train.csv)'             )
parser.add_argument('-TT'   ,'--traintext'  ,default = 'text'           ,   help='The name of the text column (Training set). (default : text)'             )
parser.add_argument('-TC'   ,'--traincat'   ,default = 'category'       ,   help='The name of the category column (Training set). (default : category)'     )
parser.add_argument('-S'    ,'--split'      ,default = '0.7'            ,   help='The ratio of training/testing to split the trainign set. (default : 0.7)' )
parser.add_argument('-P'    ,'--predict'    ,default = 'predict.csv'    ,   help='The name of the file to predict. (default : predict.csv)'                 )
parser.add_argument('-PT'   ,'--predicttext',default = 'text'           ,   help='The name of the text column (Predict set). (default : text)'              )
parser.add_argument('-PC'   ,'--predictcat' ,default = 'category'       ,   help='The name of the category column (Predict set). (default : category)'      )
parser.add_argument('-O'    ,'--output'     ,default = 'output.csv'     ,   help='The name of the output file. (default : output.csv)'                      )
parser.add_argument('-TSP'  ,'--trainsep'   ,default = ';'              ,   help='The csv separator for the training set. (default : ;)'                    )
parser.add_argument('-PSP'  ,'--predictsep' ,default = ';'              ,   help='The csv separator for the prediction set. (default : ;)'                  )
parser.add_argument('-F'    ,'--feature'    ,default = 'feature.txt'    ,   help='The features file. (default : features.txt)'                              )
args = parser.parse_args()



#Contains regex extressions for cleaning
CLEAN_REGEX = {
    'url'       : 'http.?://[^\s]+[\s]?'                    ,
    'username'  : '@[^\s]+[\s]?'                            ,
    'tag'       : '#[^\s]+[\s]?'                            ,
    'empty'     : 'Not Available'                           ,
    'number'    : '\s?\d+\.?\d*'                            ,
    'special'   : '[^\w+\s]'                                ,
    }

def clean_tweet(tweet_str):
    """
        Cleans a tweet from Urls, Usernames, Empty tweets, Special Characters, Numbers and Hashtags.
    """
    try :
        #Create the combined expression to eliminate all unwanted strings
        clean_ex    = '|'.join(['(?:{})'.format(x) for x in CLEAN_REGEX.values()])
        result      = re.sub(clean_ex, '', tweet_str).strip()

        #Remove character sequences
        return re.sub('([a-z])\\1+', '\\1\\1', result).strip()
    except :
        #Some times the DataFrame contains Nan
        return ''

def stem_tweet(tokens):
    """
        Stemming the process of reducing a derived word to it's original word.
    """
    #Using SnowballStemmer for english
    stemmer     = nltk.SnowballStemmer('english')
    return [stemmer.stem(x) for x in tokens]

def tokenize_tweet(tweet_str):
    """
        Tokenization.
    """
    #Using SnowballStemmer for english
    return nltk.word_tokenize(tweet_str,'english')

def init():
    """
        - Initialize the training and testing sets.
        - Cleans the data frames
        - Tokenization
        - Stemming
    """
    pol_path    = os.path.join(os.getcwd(),'pol.csv')
    if not os.path.isfile(pol_path) :
        train_path  = os.path.join(os.getcwd(), args.train)

        train_df    = pd.DataFrame.from_csv(train_path,sep =  args.trainsep )
        train_df            = train_df[pd.isnull(train_df[args.traintext]) == False]
        train_df            = train_df[pd.isnull(train_df[args.traincat ]) == False]
        #Clean the data frames
        print('Cleaning training data sets started!')
        train_df[args.traintext]    = train_df[args.traintext].apply(clean_tweet)
        #Remove empty entries
        train_df                    = train_df[train_df[args.traintext  ]!='']
        train_df                    = train_df[train_df[args.traincat   ]!='']
        print('Cleaning training data sets done!')

        #Stemming and tokenization
        print('Tokenizing and stemming training data sets started!')
        train_df['tokens']  = train_df[args.traintext].apply(tokenize_tweet   )
        train_df['stem']    = train_df['tokens'].apply(stem_tweet       )
        print('Tokenizing and stemming training data sets done!')

        #Counting the words occurences and calculating the polarization
        count_df            = words_occ(train_df)
    else :
        train_df,count_df   = None, None
    words_df,pol_df     = words_pol(train_df,count_df)

    train_args                                          = get_train_data(words_df)
    precision, recall, accuracy, f1,model,labels        = train(*train_args)
    tokens                                              = words_df.columns[1:]

    print ('Prediction accuracy : {}'.format(accuracy))
    return model,tokens,labels

def words_occ(df,column = 'stem'):
    """
        Get the occurences of words in a given data frame
    """
    print ('Counting tokens started!')
    #Create the counter
    words_counter   = Counter()
    for x in df[column]:
        words_counter.update(x)

    #Download the stop wrods
    try :
        nltk.download('stopwords')
    except :
        pass

    #Remove the stop words, excluding not
    stop_words      = nltk.corpus.stopwords.words('english')
    stop_words.remove('not')
    for x in stop_words :
        if 'n\'t' not in x :
            del words_counter[x]

    #Return the df indexed by words, sorted buy count
    df = pd.DataFrame([[k,words_counter[k]] for k in words_counter],columns=['word','count']).set_index('word').sort_values(by = 'count',ascending = False)
    #Remove words with less than three occurences
    df = df[df['count']>3]
    print ('Counting tokens done!')

    return df

def words_pol(df_tweets,df_words):
    """
        Creates a data frame contining the occurence of each word in df_words for every tweet in df_tweets
    """
    print ('Calculating words polarization started!')
    loaded      = False
    pol_path    = os.path.join(os.getcwd(), 'pol.csv'      )
    try :
        df          = pd.DataFrame.from_csv(pol_path)
    except Exception as e  :
        df = pd.DataFrame(
            [
                [row[args.traincat]]+[row['stem'].count(word) for word in df_words.index] for id_,row in df_tweets.iterrows()],
            columns = [args.traincat]+list(df_words.index)
        )
        df.to_csv(pol_path)

    #Group the rows by emotion, and sum the groups
    #This produces the total occurences of a word for each emotion
    gdf = df.groupby([args.traincat]).sum()

    print ('Calculating words polarization done!')

    #Save feature file
    with open(args.feature,'w') as f :
        df.groupby(args.traincat).sum().T.to_string(f)

    #Flip and return the DataFrame
    return df,gdf.T

def train(train_x,test_x,train_y,test_y,classifier=BernoulliNB()):
    '''
        Train the classifier and test agianst the test data
    '''
    print('Training started!')
    model       = classifier.fit(train_x, train_y)
    predictions = model.predict(test_x)
    labels      = sorted(list(set(train_y)))
    precision   = precision_score(test_y, predictions, average=None, pos_label=None, labels=labels)
    recall      = recall_score(test_y, predictions, average=None, pos_label=None, labels=labels)
    accuracy    = accuracy_score(test_y, predictions)
    f1          = f1_score(test_y, predictions, average=None, pos_label=None, labels=labels)
    print('Training done!')

    return precision, recall, accuracy, f1,model,labels

def get_train_data(words_df):
    '''
        Split the training data into test and train data
    '''
    return  train_test_split(words_df.iloc[:, 1:].values, words_df.iloc[:, 0].values,train_size=float(args.split),test_size = 1-float(args.split), stratify=words_df.iloc[:, 0].values,random_state=123)

def predict(model, tweet,tokens):
    '''
        Predicts the emotion of the tweet using the trained model
    '''
    stemmed = stem_tweet(tokenize_tweet(clean_tweet(tweet)))
    mask    =  [stemmed.count(token) for token in tokens ]
    return list(model.predict([mask]))+ list(model.predict_proba([mask])[0])

def predict_set(m,t,l):
    '''
        Predict the category of a given data set
    '''
    predict_path            = os.path.join(os.getcwd(), args.predict)
    df                      = pd.DataFrame.from_csv(predict_path,sep = args.predictsep)
    l                       = [args.predictcat]+l
    result                  = []
    for txt in df[args.predicttext].values :
        p = predict(m,txt,t)
        result.append(p)

    for col in  l :
        df[col]     = [p[l.index(col)] for p in result]

    df.to_csv(args.output)


if __name__ == '__main__':
    m,t,l = init()
    predict_set(m,t,l)
