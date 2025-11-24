import nltk
import joblib
import string
import numpy as np
import pandas as pd
import seaborn as sns 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix



dataMessages = pd.read_excel('Dataset1.xlsx')
dataMessages.head(5)
"""
labels = ['Classe 0 : messages non haineux', 'Classe 1 : messages haineux']

sns.countplot(x='Decision', data=dataMessages, hue='Decision', legend=False)

sns.set_style('whitegrid')
plt.xlabel('Classe')
plt.ylabel('Nombre de messages')
plt.title('Nombre de messages en foncion de la classe')
plt.legend(labels)

plt.show()"""

def cleanData(sentences):
    sentences = sentences.lower()
    sentences = sentences.translate(str.maketrans('','',string.punctuation)) # Suppression des caracteres spéciaux et la ponctuation
   
    words = set(stopwords.words('french')) # creation d'un ensemble de stop word de la langue francaise telles que (et, la , et etc...)
    
    sentences = ' '.join(word for word in sentences.split() if word not in words)

    return sentences

dataMessages['Message'] = dataMessages['Message'].apply(cleanData)

def convertToDataFrame (messageList):

    if isinstance(messageList, list):
        messageList = pd.DataFrame(messageList, columns=['Message'])

    elif isinstance(messageList, str):
        messageList = [{'Message' : messageList}]
        messageList = pd.DataFrame(messageList)

    else:
        pass

    return messageList

def vectorise (dataFrame, colName, vectorizer=None):
    messages = dataFrame[colName]

    """
        methode Bag-of-Words:
            methode utilisée pour representer les donneés textuelles sous
            forme numériques.
        Idée : 
            compter la fréquence d'apparition des mots et leur presence
            dans chaque phrase sans tenir compte de leur ordre.
    """
    if vectorizer is None:
        vectorizer = CountVectorizer()
        messagesVectorised = vectorizer.fit_transform(messages)
    else:
        messagesVectorised = vectorizer.transform(messages)

    """
    On pouvait proceder en deux etape :
    vectorizer.fit(messages) #Apprentissage du vocabulaire a partir des phrases contenues dans 'Messages'.
    messagesVectorised = vectorizer.transform(messages) #Transforme chaque phrase en vecteur numerique en comptant le nombre de mot du vocabulaire appris avec la methode vectorizer.fit()
    """ 
    return messagesVectorised, vectorizer


messagesVectorised, vectorizer = vectorise(dataMessages, 'Message')


decision = dataMessages['Decision']

X_train, X_test, y_train, y_test = train_test_split(messagesVectorised, decision, test_size=0.25, random_state=15)

"""
    Sépartion des données d'entrainement et de test
    75% ==> données d"entrainement
    25% ==> données de test
"""

model = LogisticRegression(class_weight='balanced', tol=1e-5)
#penalty=None, dual=False, tol=1e-5, 

model.fit(X_train, y_train)

performance = model.score(X_train, y_train) #performances
print(performance)

y_predict = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)

confusionMatrix = confusion_matrix(y_test, y_predict)
"""
plt.figure(figsize=(10, 6))

plt.figure(figsize=(8, 6))

sns.heatmap(confusionMatrix, annot=True, cmap='coolwarm')
plt.xlabel('Prédictions')
plt.ylabel('Données reelles')
plt.title('matrice de confusion')

plt.show()"""

print(f"accuracy = {accuracy}")



joblib.dump(model, 'logistic_regression_model.pkl')


def classifyData (data):
    result = []
    data = cleanData(data)
    data = convertToDataFrame(data)
    data, _= vectorise(data, 'Message', vectorizer)
    #model = joblib.load('logistic_regression_model.pkl')
    prediction = model.predict(data)
    return prediction
data = "La police tue"
decision = classifyData(data)
print(decision)
