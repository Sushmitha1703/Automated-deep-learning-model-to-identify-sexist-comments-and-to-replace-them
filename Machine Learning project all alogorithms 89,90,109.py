#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm.notebook import tqdm_notebook


# In[2]:


import nltk
from nltk.corpus import stopwords


# In[3]:


train = pd.read_csv('sexismdataset.csv')


# In[1]:


train.shape


# In[5]:


train.head()


# In[6]:


train.info()


# In[7]:


train.isnull().sum()


# In[8]:


train['oh_label'].value_counts()


# In[9]:


train['oh_label'].nunique()


# In[10]:


train.dropna()


# In[11]:


train['oh_label'].dtype


# In[12]:


train.isnull().sum()


# In[13]:


train.dropna(inplace= True)


# In[14]:


train.isnull().sum()


# In[15]:


train


# In[16]:


train['oh_label']=train['oh_label'].astype(int)


# In[17]:


train


# In[18]:


train.drop(['index','id','Annotation'],axis=1)


# In[19]:


train[train.duplicated()]


# In[20]:


train


# In[21]:


df=train.drop(['index','id','Annotation'],axis=1)


# In[22]:


df


# In[23]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[24]:


from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()
     


# In[25]:


import nltk
nltk.download('omw-1.4')
def clean(text):

  cleanr = re.compile('<[^>]*>')           # remove html
  cleantext = re.sub(cleanr, ' ', text)

  cleantext = re.sub("[-]", " " , cleantext)   # remove - sign

  cleantext = re.sub("[^A-Za-z ]", " " , cleantext)  # remove evey character except alphabet
  cleantext = cleantext.lower()

  words = nltk.tokenize.word_tokenize(cleantext)
  words_new = [i for i in words if i not in stop_words]

  w = [lemmatizer.lemmatize(word) for word in words_new if len(word)>2]

  return ' '.join(w)


# In[26]:


df['Text'] = tqdm_notebook(df['Text'].apply(clean))


# In[27]:


df


# In[ ]:





# In[28]:


from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(df['Text'])
sequences = tokenizer.texts_to_sequences(df['Text'])
padded_sequences = pad_sequences(sequences, maxlen=20)


# In[29]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(padded_sequences, df['oh_label'], test_size = 0.2, stratify=df['oh_label'], random_state = 42)


# In[ ]:





# In[30]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from keras.layers import Dense,LSTM, SpatialDropout1D, Embedding
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow import keras


# In[31]:


model = Sequential()
model.add(Embedding(input_dim=30000, output_dim=20, input_length=20))
model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 64

history = model.fit(X1_train, y1_train,validation_data = (X1_test,y1_test), epochs=epochs, batch_size=batch_size)


# In[53]:


y_pred_lstm=model.predict(X1_test)
y_pred_lstm = (y_pred_lstm > 0.5).astype('int32')


# In[56]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc1 = roc_auc_score(y1_test, y_pred_lstm)
acc1=accuracy_score(y1_test,y_pred_lstm)
precision1 = precision_score(y1_test, y_pred_lstm)
recall1 = recall_score(y1_test, y_pred_lstm)
f11 = f1_score(y1_test, y_pred_lstm)
print("AUC:", auc)


# In[57]:


print("Accuracy:", acc1)
print("Precision:", precision1)
print("Recall:", recall1)
print("F1 Score:", f11)
print("AUC Score:", auc1)


# In[37]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# # BiLSTM

# In[58]:


from keras.layers import Bidirectional


# In[59]:


import tensorflow as tf
import tensorflow.keras.layers as layers

model3 = Sequential()
model3.add(Embedding(input_dim=30000, output_dim=20, input_length=20))
model3.add(Bidirectional(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2)))
model3.add(Dense(units=1, activation='sigmoid'))
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 64

history3 = model3.fit(X1_train, y1_train,validation_data = (X1_test,y1_test), epochs=epochs, batch_size=batch_size)


# In[60]:


y_pred_bi=model3.predict(X1_test)
y_pred_bi=(y_pred_bi > 0.5).astype('int32')


# In[128]:


acc2 = history3.history['accuracy']
val_acc2 = history3.history['val_accuracy']
loss2 = history3.history['loss']
val_loss2 = history3.history['val_loss']
epochs = range(1, len(acc2) + 1)
plt.plot(epochs, acc2, 'r', label='Training accuracy')
plt.plot(epochs, val_acc2, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss2, 'r', label='Training loss')
plt.plot(epochs, val_loss2, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[62]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc2 = roc_auc_score(y1_test, y_pred_bi)
acc2=accuracy_score(y1_test,y_pred_bi)
precision2 = precision_score(y1_test, y_pred_bi)
recall2 = recall_score(y1_test, y_pred_bi)
f12 = f1_score(y1_test, y_pred_bi)
print("AUC:", auc)


# In[63]:


print("Accuracy:", acc2)
print("Precision:", precision2)
print("Recall:", recall2)
print("F1 Score:", f12)
print("AUC Score:", auc2)


# # logistic Regression

# In[45]:


from sklearn.linear_model import LogisticRegression 


# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=75)
X1 = tfidf.fit_transform(df['Text']).toarray()
y1 = df['oh_label']


# In[47]:


from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X1, y1, test_size = 0.3, stratify=y1, random_state = 42)


# In[48]:


logreg = LogisticRegression(solver='liblinear') 
logreg.fit(X2_train,y2_train) 
y_pred=logreg.predict(X2_test) 


# In[49]:


y_pred


# In[50]:


from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.metrics import f1_score
accuracy = accuracy_score(y2_test, y_pred)
r2 = r2_score(y2_test,y_pred)
auc = roc_auc_score(y2_test,y_pred)
print('Accuracy of logistic regression classifier: {:.2f}%'.format(accuracy*100))
print('auc score of logistic regression classifier: {:.2f}'.format(auc))


# In[51]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc = roc_auc_score(y2_test, y_pred)
acc=accuracy_score(y2_test,y_pred)
precision = precision_score(y2_test, y_pred)
recall = recall_score(y2_test, y_pred)
f1 = f1_score(y2_test, y_pred)
print("AUC:", auc)


# In[52]:


print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC Score:", auc)


# In[73]:


len(X2_train)


# In[74]:


from sklearn import metrics 
cnf_matrix1 = metrics.confusion_matrix(y2_test, y_pred) 
cnf_matrix1 


# In[75]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score
cm_display = ConfusionMatrixDisplay(confusion_matrix = cnf_matrix1)
cm_display.plot()
plt.show()


# In[ ]:





# # Neural Networks

# In[64]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=75)
X1 = tfidf.fit_transform(df['Text']).toarray()
y1 = df['oh_label']


# In[65]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X1, y1, test_size = 0.2, stratify=y1, random_state = 42)


# In[66]:


from keras.utils.np_utils import to_categorical
y3_train=to_categorical(y1_train, num_classes = 2, dtype='float32')
y3_test=to_categorical(y1_test, num_classes = 2, dtype='float32')


# In[67]:


import tensorflow as tf
import tensorflow.keras.layers as layers
model2 = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(2,activation='sigmoid')
])


model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history2=model2.fit(X3_train, y3_train, epochs=10, validation_data=(X3_test, y3_test))


# In[79]:


acc1 = history2.history['accuracy']
val_acc1 = history2.history['val_accuracy']
loss1 = history2.history['loss']
val_loss1 = history2.history['val_loss']
epochs = range(1, len(acc1) + 1)
plt.plot(epochs, acc1, 'r', label='Training accuracy')
plt.plot(epochs, val_acc1, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss1, 'r', label='Training loss')
plt.plot(epochs, val_loss1, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

loss, accuracy = model2.evaluate(X3_test, y3_test)
print('Accuracy: {:.2f}%'.format(accuracy*100))

y_pred_prob = model2.predict(X3_test)


y_pred_prob = (y_pred_prob > 0.5).astype('int32')

auc_score = roc_auc_score(y3_test, y_pred_prob)
print('AUC score: {:.2f}'.format(auc_score))


# In[77]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc3 = roc_auc_score(y3_test, y_pred_prob)
acc3=accuracy_score(y3_test,y_pred_prob)
precision3 = precision_score(y3_test, y_pred_prob,average='micro')
recall3 = recall_score(y3_test, y_pred_prob,average='micro')
f13 = f1_score(y3_test, y_pred_prob,average='micro')
print("AUC:", auc3)


# In[78]:


print("Accuracy:", acc3)
print("Precision:", precision3)
print("Recall:", recall3)
print("F1 Score:", f13)
print("AUC Score:", auc3)


# # Decision Tree Classifier

# In[81]:


from sklearn.tree import DecisionTreeClassifier


# In[83]:


clf = DecisionTreeClassifier(max_depth =4, random_state = 42)
clf.fit(X2_train, y2_train)
pred_decision_tree = clf.predict(X2_test)


# In[84]:


from sklearn import tree
import matplotlib.pyplot as plt
labels=y1.unique().astype(str).tolist()
plt.figure(figsize=(50,40))

a = tree.plot_tree(clf,
                   
                   class_names=labels,

                   rounded = True,

                   filled = True,

                   fontsize=24)

plt.show()


# In[85]:


cnf_matrix2 = confusion_matrix(y2_test, pred_decision_tree )
print(cnf_matrix2)


# In[87]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_display = ConfusionMatrixDisplay(confusion_matrix = cnf_matrix2)
cm_display.plot()
plt.show()


# In[98]:


from sklearn.metrics import accuracy_score, roc_auc_score
accuracy4 = accuracy_score(y2_test, pred_decision_tree)
auc4 = roc_auc_score(y2_test, pred_decision_tree)
print('Accuracy of decision tree classifier: {:.2f}%'.format(accuracy4*100))
print('AUC score of decision tree classifier: {:.2f}%'.format(auc4))


# In[89]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc4 = roc_auc_score(y2_test, pred_decision_tree)
acc34=accuracy_score(y2_test, pred_decision_tree)
precision4 = precision_score(y2_test, pred_decision_tree)
recall4 = recall_score(y2_test, pred_decision_tree)
f14 = f1_score(y2_test, pred_decision_tree)
print("AUC:", auc4)


# In[90]:


print("Accuracy:", accuracy4)
print("Precision:", precision4)
print("Recall:", recall4)
print("F1 Score:", f14)
print("AUC Score:", auc4)


# # Random Forest Classifier

# In[91]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)

classifier_rf.fit(X2_train, y2_train)
classifier_rf.oob_score_


# In[92]:


Y_pred = classifier_rf.predict(X2_test)


# In[93]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print('Random Forest Classifier:')
from sklearn.metrics import accuracy_score, f1_score
print('Accuracy score:', round(accuracy_score(y2_test, Y_pred) * 100, 2))
print('F1 score:', round(f1_score(y2_test, Y_pred, average='weighted') * 100, 2))


# In[94]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(80,60))
plot_tree(classifier_rf.estimators_[5],class_names=labels,filled=True);


# In[99]:


from sklearn.metrics import accuracy_score, roc_auc_score
accuracy5 = accuracy_score(y2_test, Y_pred)
auc5 = roc_auc_score(y2_test,Y_pred)
print('Accuracy: {:.2f}%'.format(accuracy5*100))
print('AUC: {:.2f}%'.format(auc5))


# In[96]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc5 = roc_auc_score(y2_test, Y_pred)
acc5=accuracy_score(y2_test, Y_pred)
precision5 = precision_score(y2_test,Y_pred)
recall5 = recall_score(y2_test,Y_pred)
f15 = f1_score(y2_test, Y_pred)
print("AUC:", auc5)


# In[97]:


print("Accuracy:", accuracy5)
print("Precision:", precision5)
print("Recall:", recall5)
print("F1 Score:", f15)
print("AUC Score:", auc5)


# # AdaBoost Classifier

# In[100]:


from sklearn.ensemble import AdaBoostClassifier


# In[101]:


clf2 = AdaBoostClassifier(n_estimators=100)
clf2 = clf2.fit(X2_train, y2_train)
y_pred_ada = clf.predict(X2_test)
accuracy6 = accuracy_score(y2_test, y_pred)
auc6 = roc_auc_score(y2_test, y_pred)
print("accuracy is: ",accuracy)
print("AUC:",auc)


# In[102]:


from sklearn import metrics 
cnf_matrix3 = metrics.confusion_matrix(y2_test, y_pred_ada) 
cnf_matrix3


# In[103]:


cm_display = ConfusionMatrixDisplay(confusion_matrix = cnf_matrix3)
cm_display.plot()
plt.show()


# In[104]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc6 = roc_auc_score(y2_test, y_pred_ada)
acc6=accuracy_score(y2_test,  y_pred_ada)
precision6 = precision_score(y2_test, y_pred_ada)
recall6 = recall_score(y2_test, y_pred_ada)
f16 = f1_score(y2_test, y_pred_ada)
print("AUC:", auc6)


# In[105]:


print("Accuracy:", accuracy6)
print("Precision:", precision6)
print("Recall:", recall6)
print("F1 Score:", f16)
print("AUC Score:", auc6)


# In[ ]:





# # XGBoost Classfier

# In[107]:


import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=X2_train,label=y2_train,enable_categorical=True)


# In[108]:


from xgboost import XGBClassifier
params = {
            'objective':'multi:softmax',
            'max_depth': 4,
            'num_class': 2,
            'alpha': 10,
            'learning_rate': 0.01,
            'n_estimators':100
        }
            

xgb_clf = XGBClassifier(**params)

xgb_clf.fit(X2_train, y2_train)


# In[110]:


y_pred_xg = xgb_clf.predict(X2_test)
from sklearn.metrics import accuracy_score
auc7 = roc_auc_score(y2_test,y_pred_xg)
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y2_test, y_pred_xg)))
print("AUC:",auc7)


# In[111]:


from xgboost import cv

params = {'objective':'multi:softmax','colsample_bytree': 0.3,'learning_rate': 0.01, 'num_class': 2,
                'max_depth': 4, 'alpha': 10}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)


# In[112]:


xgb_cv.head()


# In[113]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc7 = roc_auc_score(y2_test, y_pred_xg)
acc7=accuracy_score(y2_test,  y_pred_xg)
precision7 = precision_score(y2_test, y_pred_xg)
recall7 = recall_score(y2_test, y_pred_xg)
f17 = f1_score(y2_test, y_pred_xg)
print("AUC:", auc7)


# In[115]:


print("Accuracy:", acc7)
print("Precision:", precision7)
print("Recall:", recall7)
print("F1 Score:", f17)
print("AUC Score:", auc7)


# # KNN Classifier

# In[116]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X2_train, y2_train)

y_pred_knn = classifier.predict(X2_test)


# In[117]:


cnf = confusion_matrix(y2_test, y_pred_knn)
cnf


# In[118]:


import seaborn as sns
fig, ax2 = plt.subplots(sharex = True)
sns.heatmap(pd.DataFrame(cnf), annot=True, cmap="YlGnBu" ,fmt='g') 
ax2.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label')


# In[119]:


print("Accuracy:",metrics.accuracy_score(y2_test, y_pred_knn)) 
print("AUC:",roc_auc_score(y2_test,y_pred_knn))


# In[120]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc8 = roc_auc_score(y2_test,y_pred_knn)
acc8=accuracy_score(y2_test,y_pred_knn)
precision8 = precision_score(y2_test,y_pred_knn)
recall8 = recall_score(y2_test,y_pred_knn)
f18 = f1_score(y2_test,y_pred_knn)
print("AUC:", auc8)


# In[121]:


print("Accuracy:", acc8)
print("Precision:", precision8)
print("Recall:", recall8)
print("F1 Score:", f18)
print("AUC Score:", auc8)


# # Gaussian Nb

# In[122]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


# In[123]:


classifier = GaussianNB();
classifier.fit(X2_train, y2_train)


# In[124]:


y_pred_gnb = classifier.predict(X2_test)
cm = confusion_matrix(y2_test, y_pred_gnb)
cm


# In[125]:


print("Accuracy:",metrics.accuracy_score(y2_test, y_pred_gnb)) 
print("AUC:",roc_auc_score(y2_test,y_pred_gnb))


# In[126]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
auc9 = roc_auc_score(y2_test,y_pred_gnb)
acc9=accuracy_score(y2_test,y_pred_gnb)
precision9 = precision_score(y2_test,y_pred_gnb)
recall9 = recall_score(y2_test,y_pred_gnb)
f19 = f1_score(y2_test,y_pred_gnb)
print("AUC:", auc9)


# In[127]:


print("Accuracy:", acc9)
print("Precision:", precision9)
print("Recall:", recall9)
print("F1 Score:", f19)
print("AUC Score:", auc9)


# In[ ]:




