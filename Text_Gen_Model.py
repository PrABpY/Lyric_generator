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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.saving import load_model
from livelossplot import PlotLossesKeras

# Config
pd.set_option('max_colwidth', 999)
print("pass")
tf.config.list_physical_devices('GPU')
df = pd.read_csv('data_lyrics.csv')

df['textToken'] = df['Lyrics'].apply(lambda text: word_tokenize(text,engine = 'newmm',keep_whitespace = False))
df['textToken'] = df['textToken'].apply(lambda text: ' '.join(text))

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

tokenizer.sequences_to_texts(sequencesTokens[:7])
maxLength = max([len(x) for x in sequencesTokens])
sequencesTokens = pad_sequences(sequencesTokens,maxlen = maxLength,padding = 'pre')

features = sequencesTokens[:, :-1]

labels = sequencesTokens[:, -1]

labels = to_categorical(labels,num_classes = totalWords)

model = Sequential()
model.add(Embedding(totalWords, output_dim = 128, input_length = features.shape[1]))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(totalWords, activation = 'softmax'))
model.summary()

EPOCHS = 1000
RATE = 0.0001
opt = Adam(learning_rate = RATE)
model.compile(optimizer = opt,loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(features,labels,epochs = EPOCHS)

scores = model.evaluate(features,labels,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


model.save('Textgen_model_LSTM.keras')
file_path = "model/Textgen_"+"gen_LSTM"+"_.h5"
model.save(file_path)
plt.figure(figsize = (10, 6))
plt.plot(history.history['accuracy'],c = 'blue',label = 'Accuracy')
plt.plot(history.history['loss'],c = 'red',label = 'Loss')
plt.legend(frameon = True, facecolor = 'white')
plt.title('Model training process')
plt.xlabel('Epochs')
plt.show()







