# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pythainlp import word_tokenize
#from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


# pd.set_option('max_colwidth', 999)
print("pass__")
tf.config.list_physical_devices('GPU')

df = pd.read_csv('Dataset/data_lyrics.csv')
df.head(5)

df['textToken'] = df['Lyrics'].apply(lambda text: word_tokenize(text,engine = 'newmm',keep_whitespace = False))
df.head(5)

df['textToken'] = df['textToken'].apply(lambda text: ' '.join(text))
df.head(5)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['textToken'])
totalWords = len(tokenizer.word_index) + 1
print(f'Total unique word: {totalWords}')

sequencesTokens = []

for token in df['textToken']:
  listToken = tokenizer.texts_to_sequences([token])[0]
  for i in range(1, len(listToken)):
    sequenceToken = listToken[:i + 1]
    sequencesTokens.append(sequenceToken)

sequencesTokens[:7]
tokenizer.sequences_to_texts(sequencesTokens[:7])
maxLength = max([len(x) for x in sequencesTokens])
sequencesTokens = pad_sequences(sequencesTokens,maxlen = maxLength,padding = 'pre')
sequencesTokens[:7]

features = sequencesTokens[:, :-1]
features[:7]
labels = sequencesTokens[:, -1]
labels[:7]
labels = to_categorical(labels,num_classes = totalWords)
labels[0]

print(list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(np.where(labels[0] == 1)[0])])


model = load_model('model/Textgen_ver_26_.h5')

def generateQuotation(text, numWords):
  textOut = ' '.join(word_tokenize(text))
  wordCount = 0
  while wordCount < numWords:
    textList = tokenizer.texts_to_sequences([textOut])[0]
    textList = pad_sequences([textList],maxlen = maxLength - 1,padding = 'pre')
    predIndex = model.predict(textList, verbose = 0).argmax()
    for word, index in tokenizer.word_index.items():
      if index == predIndex:
        wordNext = word
        print(wordNext,end="")
        break
    textOut += ' ' + wordNext
    wordCount += 1

  file1 = open('lyrics/lyric_'+text+'.txt', 'w')

  file1.write(textOut)

  # Closing file
  file1.close()


texts = ["รัก"]

for text in texts:
  print("=="*10+text+"=="*10)
  generateQuotation(text,600)
  print()





