import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
from tcn import TCN, tcn_full_summary
from keras.layers import Dense, Bidirectional, Flatten, ZeroPadding2D, Reshape
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import GRU

from keras.models import Sequential, Model
from keras.utils import np_utils

from keras.optimizers import Adam,RMSprop,SGD


MODEL_DIR = './model2'

def save_weights(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def load_weights(epoch, model):
    model.load_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def build_model(batch_size, seq_len, vocab_size):
    #model 3(GRU)
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))
    for i in range(3):
         model.add(GRU(256, return_sequences=True, stateful=True))
         model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))


    return model

if __name__ == '__main__':
    model = build_model(16, 64, 50)
    model.summary()
