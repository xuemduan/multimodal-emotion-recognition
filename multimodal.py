import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

import pickle 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

import re
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


def load_data():

    info = pd.read_csv('data/AnotherMissOh/script_image_index2.csv')
    frame_id = info['frame_id'].tolist()
    character = info['character'].tolist()
    character_index = info['character_index'].tolist()
    script = info['script'].tolist()

    # idx
    idx = []
    for f, c, i in zip(frame_id, character, character_index):
        idx.append(f+'_'+c+'_'+str(i))

    # text
    texts = [' '.join(clean_text(text)) for text in script]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
 
    text_sequence= np.array(tokenizer.texts_to_sequences(texts))
    X_S = pad_sequences(text_sequence, maxlen=32)
    index_of_words = tokenizer.word_index

    vocab_size = len(index_of_words) + 1

    # image
    feature_path = 'data/AnotherMissOh/emotion_images/cache/emotion_images.pkl'
    label_path = 'data/AnotherMissOh/emotion_label.pkl'

    with open(feature_path, 'rb') as f:
        feature = pickle.load(f)

    with open(label_path, 'rb') as f:
        label = pickle.load(f)

    label_names = set(label.values())
    label_index = { k:v for v, k in enumerate(label_names) } 

    X_I = np.array([ feature[i] for i in idx ])
    Y = np.array([ label_index[label[i]] for i in idx ])

    with open('index.pkl', 'rb') as f:
        index = pickle.load(f)
        train_index = index['train']
        test_index = index['test']

    return X_S, X_I, Y, train_index, test_index, vocab_size

def clean_text(data):
   
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
   
    data = word_tokenize(data)
   
    return data

def random_forest(X, Y, XT, YT):

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, Y)

    YPred = model.predict(XT)
    print(accuracy_score(YT, YPred))
    print(classification_report(YT, YPred))

def nn_text(X, Y, XT, YT, vocab_size, batch_size=32, num_epochs=10, num_classes=7):

    # text
    embedd_matrix = np.loadtxt('embedd_matrix_3474.txt')
    embed_layer = keras.layers.Embedding(
        vocab_size,
        300,
        input_length = 32,
        weights = [embedd_matrix],
        trainable = False,
        name = 'Embedding'
    )

    text_inputs = keras.Input(shape=(32))
    text_x = embed_layer(text_inputs)
    text_x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256), name='Text')(text_x)
    text_x = keras.layers.Dropout(0.1)(text_x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name='Output')(text_x)
    model = keras.Model(inputs=text_inputs, outputs=outputs)
    model.summary()

    model.compile(
        optimizer='Adam',
        loss="sparse_categorical_crossentropy",
        metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc")
        ],
    )

    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='model/multi_best_text',
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True,
        mode='max',
        )
    ]

    history1 = model.fit(X, Y, batch_size=batch_size, epochs=num_epochs, validation_data=(XT,YT), callbacks=callbacks)

    del model 

    model = keras.models.load_model('model/multi_best_text')
    prediction1 = model.predict(X)
    prediction2 = model.predict(XT)

    return prediction1, prediction2

def nn_image(X, Y, XT, YT, batch_size=32, num_epochs=40, num_classes=7):

    # image
    image_inputs = keras.Input(shape=(3, 2048))
    image_x = keras.layers.Flatten()(image_inputs)
    image_x = keras.layers.Dropout(0.1)(image_x)
    image_x = keras.layers.Dense(256, activation='relu', name='Image')(image_x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name='Output')(image_x)

    model = keras.Model(inputs=image_inputs, outputs=outputs)
    model.summary()

    model.compile(
        optimizer='Adam',
        loss="sparse_categorical_crossentropy",
        metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc")
        ],
    )

    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='model/multi_best_image',
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True,
        mode='max',
        )
    ]

    history1 = model.fit(X, Y, batch_size=batch_size, epochs=num_epochs, validation_data=(XT,YT), callbacks=callbacks)

    del model 

    model = keras.models.load_model('model/multi_best_image')
    prediction1 = model.predict(X)
    prediction2 = model.predict(XT)

    return prediction1, prediction2

def nn(X, Y, XT, YT, vocab_size, batch_size=32, num_epochs=40, num_classes=7):

    # image
    image_inputs = keras.Input(shape=(3, 2048))
    image_x = keras.layers.Flatten()(image_inputs)
    image_x = keras.layers.Dropout(0.1)(image_x)
    image_x = keras.layers.Dense(256, activation='relu', name='Image')(image_x)

    # text
    embedd_matrix = np.loadtxt('embedd_matrix_3474.txt')
    embed_layer = keras.layers.Embedding(
        vocab_size,
        300,
        input_length = 32,
        weights = [embedd_matrix],
        trainable = False,
        name = 'Embedding'
    )

    text_inputs = keras.Input(shape=(32))
    text_x = embed_layer(text_inputs)
    text_x = keras.layers.Dropout(0.1)(text_x)
    text_x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256), name='Text')(text_x)

    x = tf.keras.layers.concatenate([image_x, text_x], axis=1)
    x = keras.layers.Dense(512, activation="relu", name='combination')(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name='Output')(x)
    model = keras.Model(inputs=[image_inputs, text_inputs], outputs=outputs)
    model.summary()

    model.compile(
        optimizer='Adam',
        loss="sparse_categorical_crossentropy",
        metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc")
        ],
    )

    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='model/multi_best_all',
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True,
        mode='max',
        )
    ]

    history = model.fit(X, Y, batch_size=batch_size, epochs=num_epochs, validation_data=(XT,YT), callbacks=callbacks)

def main():

    X_S, X_I, Y, train_index, test_index, vocab_size = load_data()
    
    # end2end fusion
    XTrain, XTest, YTrain, YTest = [X_I[train_index],X_S[train_index]], [X_I[test_index],X_S[test_index]], Y[train_index], Y[test_index]
    nn(XTrain, YTrain, XTest, YTest, vocab_size)
    
    # decision fusion 
    P1, P2 = nn_text(X_S[train_index], Y[train_index], X_S[test_index], Y[test_index], vocab_size)
    P3, P4 = nn_image(X_I[train_index], Y[train_index], X_I[test_index], Y[test_index])
    X = np.concatenate([P1,P3], axis=1)
    XT = np.concatenate([P2,P4], axis=1)
    random_forest(X, Y[train_index], XT, Y[test_index])

if __name__ == '__main__':
    main()
