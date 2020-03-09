import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import keras.callbacks
import re
from keras.models import load_model
import pickle

class History(keras.callbacks.Callback):
    #will contain all loss values for every epoch run
    def on_train_begin(self, logs={}):
        self.losses=[]
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        with open('256lstm40_losses.pickle', 'wb') as handle:
            pickle.dump(self.losses, handle)


    #save five models to use later to show text gen
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 20 == 0:
            self.model.save("256LSTM40_at_epoch{}.hd5".format(epoch))
            
f = open('corpus.txt', 'r')
txt = ''
for line in f:
    txt+=line

txt = re.sub(' +', ' ', txt)
txt = txt.lower()

characters = sorted(list(set(txt)))

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

vocab_size = len(characters)
print('Number of unique characters: ', vocab_size)
print(characters)


X = []
Y = []
corp_len = len(txt)
seq_len = 40

for i in range(0, corp_len - seq_len, 1):
    seq = txt[i:i+seq_len]
    label = txt[i + seq_len]
    X.append([char_to_n[char] for char in seq])
    Y.append(char_to_n[label])
    
print('num of extracted seqs: ', len(X))

x_mod = np.reshape(X, (len(X), seq_len, 1))
x_mod = x_mod / float(len(characters))
Y_mod = to_categorical(Y)



model = load_model('256LSTM40_at_epoch0.hd5')
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define how model checkpoints are saved
# filepath = "model_weights-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history = History()
model.fit(x_mod, Y_mod, epochs=100, batch_size=128, callbacks = [history])

#final text generation
start = 10   #random row from the X array
string_mapped = list(X[start])
full_string = [n_to_char[value] for value in string_mapped]


losses = history.losses
epochs = range(len(losses))

plt.plot(epochs, losses, 'b', label='Loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('LSTM Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.show()
plt.savefig('lossplot_256lstm40.png')

with open('256lstm40.pickle', 'wb') as handle:
    pickle.dump(history, handle)

