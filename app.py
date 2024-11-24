from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(x_train.shape)
print(x_test.shape)
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10

(x_train, _), (_, labels) = cifar10.load_data()
idx = [3, 6, 25, 46, 58, 85, 93, 99, 108, 133]
clsmap = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

plt.figure(figsize=(10, 10))
for i, (img, y) in enumerate(zip(x_train[idx], labels[idx])):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(str(y[0]) + " " + clsmap[y[0]])
    plt.xticks([])
    plt.yticks([])

plt.show()
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

input_dim = (32, 32, 3)
input_img = Input(shape=input_dim)
C11 = Conv2D(64, (9, 9), strides=(2, 2), activation='relu', input_shape=input_dim)(input_img)
p12 = MaxPooling2D(pool_size=(2, 2))(C11)
bn13 = BatchNormalization()(p12)
c14 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu')(bn13)
p15 = MaxPooling2D(pool_size=(2, 2))(c14)
bn16 = BatchNormalization()(p15)
f17 = Flatten()(bn16)
do18 = Dropout(0.5)(f17)
d19 = Dense(units=256, activation='relu')(do18)
do110 = Dropout(0.2)(d19)
d111 = Dense(units=128, activation='relu')(do110)
do112 = Dropout(0.2)(d111)
output = Dense(units=10, activation='softmax')(do112)

classifier = Model(input_img, output)
opt = RMSprop(learning_rate=0.0001)
classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

classifier.summary()
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Assuming x_train, y_train, x_test, and y_test are already defined and preprocessed

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_delta=1e-4, mode='min', verbose=1)
stop_alg = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=1)

hist = classifier.fit(x_train, y_train, batch_size=100, epochs=100, validation_data=(x_test, y_test), callbacks=[reduce_lr, stop_alg], shuffle=True)

classifier.save_weights('cnn.hdf5')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], color='#785ef0')
plt.plot(hist.history['val_loss'], color='#dc267f')
plt.title('Model Loss Progress')
plt.ylabel('Binary Cross Entropy Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='upper right')
plt.show()
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()

# Assuming x_test and y_test are already defined and preprocessed
# Assuming 'classifier' is a trained model
y_hat = classifier.predict(x_test)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_true, y_pred))

# Compute and print confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Compute and print Balanced Error Rate (BER)
ber = 1 - balanced_accuracy_score(y_true, y_pred)
print('BER:', ber)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Define class map (assuming you have a dictionary mapping class indices to class names)
clsmap = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

tick_marks = np.arange(len(clsmap))
plt.xticks(tick_marks, clsmap.values(), rotation=45)
plt.yticks(tick_marks, clsmap.values())

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# Get the weights of the first convolutional layer
cnn11 = classifier.layers[1].name
w = classifier.get_layer(name=cnn11).get_weights()[0]
wshape = w.shape

# Scale the weights to the range [0, 1]
scaler = MinMaxScaler()
w_reshaped = w.reshape(-1, 1)
scaler.fit(w_reshaped)
w_scaled = scaler.transform(w_reshaped).reshape(wshape)

# Plot the filters
num_filters = wshape[-1]
num_columns = 8  # Define the number of columns for subplot grid
num_rows = int(np.ceil(num_filters / num_columns))

fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2))
fig.subplots_adjust(hspace=.25, wspace=.001)
axs = axs.ravel()

for i in range(num_filters):
    h = np.reshape(w_scaled[:, :, :, i], (wshape[0], wshape[1], wshape[2]))
    axs[i].imshow(h)
    axs[i].set_title('Filter ' + str(i))
    axs[i].axis('off')

# Turn off any unused subplots
for j in range(i + 1, len(axs)):
    axs[j].axis('off')

plt.show()
