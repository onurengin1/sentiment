from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.metrics import Precision, Recall

print(" Loading Data ")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)

x_train = sequence.pad_sequences(x_train, maxlen=80)
x_test = sequence.pad_sequences(x_test, maxlen=80)

model = Sequential()

model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',Recall(),Precision()],)

print(model.summary())
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    verbose=2, validation_data=(x_test, y_test))

loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy  : {:.4f}'.format(accuracy))
print('Precision : {:.4f}'.format(precision))
print('Recall    : {:.4f}'.format(recall))

model.save('RNN.h5')
print("model kaydedildi")