import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TensorFlow MNIST Experiment")

# Load the mnist dataset.
train_ds, test_ds = tfds.load(
  "mnist",
  split=["train", "test"],
  shuffle_files=True,
)

# Preprocess the data.
def preprocess_fn(data):
  image = tf.cast(data["image"], tf.float32) / 255
  label = data["label"]
  return (image, label)

train_ds = train_ds.map(preprocess_fn).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_fn).batch(128).prefetch(tf.data.AUTOTUNE)

# Build a simple model.
input_shape = (28, 28, 1)
num_classes = 10

model = keras.Sequential(
  [
      keras.Input(shape=input_shape),
      keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(num_classes, activation="softmax"),
  ]
)

# Compile the model.
model.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(),
  optimizer=keras.optimizers.Adam(0.001),
  metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model.
mlflow.tensorflow.autolog()

model.fit(x=train_ds, epochs=10)
score = model.evaluate(test_ds)

print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]: .2f}")

# If desire, we can log the model using MLFlow Callback for more control.
# from mlflow.tensorflow import MlflowCallback

# # Turn off autologging.
# mlflow.tensorflow.autolog(disable=True)

# with mlflow.start_run() as run:
#   model.fit(
#       x=train_ds,
#       epochs=10,
#       callbacks=[MlflowCallback(run)],
#   )