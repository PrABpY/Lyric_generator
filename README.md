# Lyrics Generator (Thai Somg)<br>
   ## Requirement<br>
     matplotlib==3.9.2
     seaborn==0.13.2
     scikit-learn==1.6.0
     tensorboard==2.10.1
     tensorboard-data-server==0.6.1
     tensorboard-plugin-wit==1.8.1
     tensorflow==2.10.0
     tensorflow-estimator==2.10.0
     tensorflow-io-gcs-filesystem==0.31.0
     tensorflow-intel==2.13.0
     keras==2.10.0
     Keras-Preprocessing==1.1.2
     
      Or
       
    $ pip install -r requirements.txt

## Import package file
   import lib and check version Tensorflow.
   ```python 
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   
   from pythainlp import word_tokenize
   
   import tensorflow as tf
   from tensorflow.keras.preprocessing.text import Tokenizer
   from tensorflow.keras.preprocessing.sequence import pad_sequences
   from tensorflow.keras.utils import to_categorical
   from tensorflow.keras.optimizers import Adam
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
   
   print(f'Tensorflow version: {tf.__version__}')
   ```
  
