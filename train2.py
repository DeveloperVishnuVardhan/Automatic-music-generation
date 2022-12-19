import os
import json
import argparse
import numpy as np
from model2 import build_model, save_weights

DATA_DIRECTORY = './data'
LOG_DIRECTORY = './logs'

BATCH_LENGTH = 16
SEQUENCE_LENGTH = 64

class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        file = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(file)

def read_batches(T, vocab_size):
    length = T.shape[0]; #129,665
    batch_chars = int(length / BATCH_LENGTH); # 8,104

    for start in range(0, batch_chars - SEQUENCE_LENGTH, SEQUENCE_LENGTH): # (0, 8040, 64)
        input = np.zeros((BATCH_LENGTH, SEQUENCE_LENGTH)) # 16X64
        output = np.zeros((BATCH_LENGTH, SEQUENCE_LENGTH, vocab_size)) # 16X64X86
        for batch_idx in range(0, BATCH_LENGTH): # (0,16)
            for i in range(0, SEQUENCE_LENGTH): #(0,64)
                input[batch_idx, i] = T[batch_chars * batch_idx + start + i] # 
                output[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield input, output

def train(text, epochs=100, save_freq=10):

    # character to index and vice-versa mappings
    charecter_to_index = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
    print("Number of unique characters: " + str(len(charecter_to_index))) #86

    with open(os.path.join(DATA_DIRECTORY, 'char_to_idx.json'), 'w') as f:
        json.dump(charecter_to_index, f)

    idx_to_char = { i: ch for (ch, i) in charecter_to_index.items() }
    vocab_size = len(charecter_to_index)

    #model_architecture
    model = build_model(BATCH_LENGTH, SEQUENCE_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    #Train data generation
    T = np.asarray([charecter_to_index[c] for c in text], dtype=np.int32) #convert complete text into numerical indices

    print("Length of text:" + str(T.size)) #129,665

    steps_per_epoch = (len(text) / BATCH_LENGTH - 1) / SEQUENCE_LENGTH  

    log = TrainLogger('training_log.csv')

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        
        losses, accuracies = [], []

        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            
            print(X);

            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))
            losses.append(loss)
            accuracies.append(acc)

        log.add_entry(np.average(losses), np.average(accuracies))

        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='input.txt', help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

    train(open(os.path.join(DATA_DIRECTORY, args.input)).read(), args.epochs, args.freq)
