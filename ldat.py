# **Dataset Implementation**

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
df = pd.read_csv('/content/drive/MyDrive/dataset.csv')


df.head()

df.isnull().sum()

df.shape

df.describe()

df["language"].value_counts ()

l=df.language.unique()

a=list(l)
type(a)
a

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

x = np.array(df["Text"])
y = np.array(df["language"])

cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)

model = MultinomialNB()
model.fit(x_train,y_train)
#model. score(x_test,y_test)

# **Accuracy,Precision,Recall Calculation**

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

y_pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1score = f1_score (y_test, y_pred, average='macro')


print (f"Accuracy = {accuracy}")
print (f"Precision = {precision}")
print (f"Recall = {recall}")
print (f"F1 Score = {f1score}")

# **Confusion Matrix**

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.set(font_scale = 0.5)

ax = sns.heatmap(conf_matrix, annot=True,fmt='d',cbar=False)

ax.set_xlabel("Predicted", labelpad=20)
ax.set_ylabel("Actual", labelpad=20)
plt.show()

# **IMAGE**

from google.colab import files
from PIL import Image

# from PIL import Image

uploaded = files.upload()

name = list(uploaded.keys())
finname = name [0]
finname

!pip install pytesseract

!sudo apt install tesseract-ocr

import pytesseract

im=Image.open(finname)
im.show()

extracted= pytesseract.image_to_string(im)
print (extracted, end=" ")


newStr = ""
for i in extracted:
  newStr+=i
newStr = newStr.replace("\n", "")
newStr

#user = input ("Enter a Text: ")
data = cv.transform([extracted]).toarray()
output = model.predict(data)
print ("predicted language: -\n",output[0])


!pip install googletrans==3.1.0a0

from googletrans import Translator

import googletrans

googletrans.LANGUAGES

# LANGUAGES
finlang = googletrans.LANGUAGES.values()
finlang = list(finlang)
print(finlang)

#*To display a dropdown of languages*

import ipywidgets as widgets
from IPython.display import display

Dropdown_ = widgets.Combobox(

    placeholder='Select..',
    options=finlang,
    description='Choose a language:',
    ensure_option=True,
    disabled=False
)

output = widgets.Output()
a = ""
def on_change(change):
   global a
   a = change['new']
  # if change['new'] > change['old']:
  #   print('bigger')
  # else:
  #   print('smaller')
Dropdown_.observe(on_change, names='value')
display (Dropdown_)



print('Language selected -', a)

trans=Translator ()

d=googletrans.LANGUAGES
key_list = list(d.keys())
val_list = list(d.values())
position = val_list.index(a)
short_name=key_list[position]
short_name
out=trans.translate(newStr,dest=short_name)
out.text
