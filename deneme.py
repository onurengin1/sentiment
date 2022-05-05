
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
from keras.metrics import Precision, Recall, accuracy
from keras_preprocessing.text import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential

data = pd.read_csv('Twitter_Data.csv')
data.dropna(axis=0, inplace=True)
data['category'] = data['category'].map({-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'})



def pad_sequence(text):
    tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, padding='post', maxlen=50)

    return X, tokenizer


X, tokenizer = pad_sequence(data['clean_text'])


y = pd.get_dummies(data['category'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


model = Sequential()
model.add(Embedding(5000, 32, input_length=50))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy', Precision(), Recall()])

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32, epochs=20, verbose=1)

loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print('')
print('Accuracy  : {:.4f}'.format(accuracy))
print('Precision : {:.4f}'.format(precision))
print('Recall    : {:.4f}'.format(recall))

