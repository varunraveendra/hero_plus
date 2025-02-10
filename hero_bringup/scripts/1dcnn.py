import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

# Ensure compatibility with TensorFlow 2.4.0
#print("TensorFlow Version:", tf.__version__)
window_size=220
# Disable GPU to ensure CPU-only computation
tf.config.set_visible_devices([], 'GPU')
data=pd.read_csv('/home/varun/catkin_ws/src/hero_plus/hero_bringup/config/10secondbehaviour.csv',header=0)
data=data.sample(frac=1)
removed_rows = data.sample(n=50)
new_df = data.drop(removed_rows.index)
test_data=removed_rows
train_data=new_df


y=test_data.iloc[:,-1]
x=test_data.iloc[:,-window_size-1:-1]#.applymap(lambda x: 1 if x < 0.8 else 0)
x_test=x.to_numpy()
y_test=y.to_numpy()

x_test=x_test.reshape(50,window_size,1)
y_test = to_categorical(y_test-1, num_classes=3)


y=train_data.iloc[:,-1]
x=train_data.iloc[:,-window_size-1:-1]#.applymap(lambda x: 1 if x < 0.8 else 0)
x_train=x.to_numpy()
y_train=y.to_numpy()

X_train=x_train.reshape(316,window_size,1)
y_train = to_categorical(y_train-1, num_classes=3)




# Sample Data Preparation
# Replace with your actual data
num_samples = 40  # Number of waveforms
time_steps = window_size  # Length of each waveform
num_classes = 3   # Number of output classes


# Build the 1D CNN Model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, 1)))
model.add(MaxPooling1D(pool_size=2))

# 2nd Convolutional Layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Flatten the output of convolutional layers
model.add(Flatten())

# Fully connected layers (Dense Layers)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization

# Output layer (Softmax activation for multi-class classification)
model.add(Dense(num_classes, activation='softmax'))

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.2)


# Plot Training History
# Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


prediction=model.predict(x_test)
y_pred = prediction.argmax(axis=1)

conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# for i in range(10):
#   prediction=model.predict(x_test[i].reshape(1, 660, 1))
#   predicted_class = np.argmax(prediction)

# # Print the predicted class
#   print(f"Predicted class: {predicted_class}")

#   # You can also print the probabilities for each class
#   print(f"Class probabilities: {prediction}")


#model.save('/home/varun/catkin_ws/src/hero_plus/hero_bringup/config/behavior10predictbin.h5')
#model.save("/home/varun/catkin_ws/src/hero_plus/hero_bringup/config/behavior10predict.keras")


























































# import tensorflow as tf
# print(tf.__version__)

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np

# def check_saved_model(model_path):
#     try:
#         # Load the model
#         print(f"Loading model from: {model_path}")
#         custom_objects = {"batch_shape": None}
#         model = load_model(model_path, custom_objects=custom_objects)

#         print("Model loaded successfully!")

#         # Display model summary
#         model.summary()

#         # Generate a dummy input tensor based on the model input shape
#         if hasattr(model, "input_shape"):
#             input_shape = model.input_shape[1:]  # Exclude batch size
#             print(f"Model input shape: {input_shape}")

#             # Create a random test input
#             dummy_input = np.random.rand(1, *input_shape).astype('float32')
#             print(f"Dummy input created with shape: {dummy_input.shape}")

#             # Perform a forward pass
#             output = model.predict(dummy_input)
#             print(f"Model forward pass successful. Output shape: {output.shape}")
#         else:
#             print("Could not determine input shape from the model. Ensure it is correctly saved.")
#     except Exception as e:
#         print("Error while loading or testing the model:")
#         print(str(e))

# # Replace 'your_model.h5' with the path to your .h5 file
# check_saved_model("/home/varun/catkin_ws/src/hero_common/hero_bringup/config/behavior_predict.h5")
