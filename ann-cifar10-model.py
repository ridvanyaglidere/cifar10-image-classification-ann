
# --- UYARILARI SUSTURMA ---
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# --- KÜTÜPHANELER ---
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# --- VERİ SETİNİ YÜKLEME ---
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(f"Index: {i}, Label: {y_train[i][0]}")
    plt.axis("off")
plt.show()

# --- NORMALİZASYON ---
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# --- ONE HOT ENCODING ---
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# --- ANN MODELİ ---
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="tanh"))
model.add(Dense(10, activation="softmax"))

model.summary()

# --- MODEL DERLEME ---
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- CALLBACKS ---
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

check_point = ModelCheckpoint(
    "ann_best_model.keras",
    monitor="val_loss",
    save_best_only=True
)

# --- MODEL EĞİTİMİ ---
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=60,
    validation_split=0.2,   # DÜZELTİLDİ
    callbacks=[early_stopping, check_point]
)

# --- MODEL TEST ---
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Loss : {test_loss}, Test Accuracy : {test_acc}")

# --- ACCURACY GRAFİĞİ ---
plt.figure()
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.title("ANN Accuracy on CIFAR-10 Dataset")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# --- LOSS GRAFİĞİ ---
plt.figure()
plt.plot(history.history["loss"], marker="o", label="Training Loss")      # DÜZELTİLDİ
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")  # DÜZELTİLDİ
plt.title("ANN Loss on CIFAR-10 Dataset")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# --- MODELİ KAYDET ---
model.save("final_cifar_ann_model.keras")

# --- MODELİ YÜKLE ---
loaded_model = load_model("final_cifar_ann_model.keras")

test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"Loaded Model -> Test Loss: {test_loss}, Test Accuracy: {test_acc}")
