import keras_preprocessing.sequence
import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# Load imdb data set with top 10,000 most frequent words
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# creates tuples
word_index = data.get_word_index()

# break tuple into key and value
word_index = {k: (v+3) for k, v in word_index.items()}

# Used to add padding to make movie review same length
word_index["<PAD>"] = 0

word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Create a reversed version of word_index to map numerical indices back to words
reversed_word_index = dict([(value, key) for(key, value) in word_index.items()])

# Preprocessing data
# Reviews must be of the same length because the input layer size is not flexible
# Reviews of length > 250 are removed and padding is add to reviews < 250
train_data = keras_preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras_preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# Convert numerical indices back to words using the reversed word index
def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text])


# Used to create and save model
"""
# Model down here
model = keras.Sequential()

# Embedding layer turns each element in the input array into separate vectors
model.add(keras.layers.Embedding(88000, 16))

# Used to reduce the dimensions
model.add(keras.layers.GlobalAvgPool1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

model.save("model.h5")
"""


def review_encode(s):
    # Add start tag
    encoded = [1]

    for word in s:
        # Add embedded word to list
        if word in word_index:
            encoded.append(word_index[word.lower()])
        # If word is not in vocabulary, add unknown tag
        else:
            encoded.append(2)
    return encoded


# Load model
model = keras.models.load_model("model.h5")

with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


