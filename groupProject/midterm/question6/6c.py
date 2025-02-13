import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

columnNames = ["Text", "Emotion"]

training = pd.read_csv("kaggleData/train.txt", sep=";", names=columnNames)

training['Clean_Text'] = training['Text'].apply(nfx.remove_userhandles)
training['Clean_Text'] = training['Clean_Text'].apply(nfx.remove_stopwords)

x_train = training['Clean_Text']
y_train = training['Emotion']

pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(x_train,y_train)

# process test data
test = pd.read_csv("kaggleData/test.txt", sep=";", names=columnNames)
test['Clean_Text'] = test['Text'].apply(nfx.remove_userhandles)
test['Clean_Text'] = test['Clean_Text'].apply(nfx.remove_stopwords)

x_test = test['Clean_Text']
y_test = test['Emotion']

print("Accuracy score: " + str(pipe_lr.score(x_test,y_test)))

analyze = pd.read_csv("kaggleData/val.txt", sep=";", names=columnNames)
analyze['Clean_Text'] = analyze['Text'].apply(nfx.remove_userhandles)
analyze['Clean_Text'] = analyze['Clean_Text'].apply(nfx.remove_stopwords)

for i in range(len(analyze['Clean_Text'])):
    print("Prediction: " + str(pipe_lr.predict([analyze['Clean_Text'][i]])) + ", Actual: " + analyze["Emotion"][i] + ", Text: " + analyze['Text'][i])