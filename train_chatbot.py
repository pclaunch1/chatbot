import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# Preprocessing the Data
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word
# Lemmatize - condense similar words to the same word; i.e. play, plays, played, playing all reduce to play
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# create training and testing data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    # get tokenized words for document
    word_patterns = doc[0]
    # lemmatize words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # check each word in word_patterns against complete list of lemmatized words.
    # array of flags, one flag for each unique lemmatized word. if word in list is in pattern, switch flag to 1
    # result in vector of flags for each word pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

# train the model
# deep neural networks model
model = Sequential()
# first layer, 128 neurons
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# neural networks commonly overfit training datasets with few examples. to combat this, add a dropout. This will randomly drop nodes in order to prevent subsequent layers from overcorrecting the errors of previous layers
model.add(Dropout(0.5))
# second layer, 64 neurons
model.add(Dense(64, activation='relu'))
# subsequent dropout
model.add(Dropout(0.5))
# third layer, number neurons equivalent to number of classes
model.add(Dense(len(train_y[0]), activation='softmax'))

#compile the model, SGD with Nesterov
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train and save
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
