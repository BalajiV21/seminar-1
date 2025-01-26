
# Importing Libraries
# Import necessary libraries for data handling, visualization, model building, and evaluation.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.manifold import TSNE

# Set random seed for reproducibility
keras.utils.set_random_seed(50)

# Load Dataset
# Load the dataset from a CSV file and display its structure.
file_path = 'Seminars_1_Group_4.csv'  
data = pd.read_csv(file_path)
print("Dataset loaded successfully:")
print(data.head())

# Data Preparation
# Separate features (X) and target (y), and encode the target variable.
target_column = 'class'  
X = data.drop(target_column, axis=1)
y = data[target_column]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Feature Selection
# Use ExtraTreesClassifier to determine feature importance and select the top N features.
clf = ExtraTreesClassifier(n_estimators=300, random_state=42)
clf.fit(X, y_encoded)

# Plot Feature Importance
feature_importances = clf.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.yticks(range(len(feature_importances)), X.columns[sorted_indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Using ExtraTreesClassifier')
plt.gca().invert_yaxis()
plt.show()

# Select Top Features
# Keep only the top N most important features.
N = 13  
top_features = X.columns[sorted_indices[:N]]
X = X[top_features]

# Data Scaling
# Standardize the feature data for better performance in neural networks.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode Target Labels for Neural Network
# Convert the encoded target labels to a one-hot encoding format.
y_label = keras.utils.to_categorical(y_encoded)

# Dimensionality Reduction
# Use t-SNE for visualizing high-dimensional data in 2D space.
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot t-SNE Visualization
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='Classes')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE Visualization')
plt.show()

# Train-Test Split
# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_label, test_size=0.3)

# Neural Network Model
# Define and build the sequential model architecture.
model = keras.models.Sequential()

# Input Layer and First Hidden Layer
model.add(keras.layers.Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
model.add(keras.layers.Dropout(0.2))  # Dropout with 20% rate

# Second Hidden Layer
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.2))  # Dropout with 20% rate

# Output Layer
model.add(keras.layers.Dense(y_label.shape[1], activation="softmax"))

# Model Summary
# Display the architecture of the model.
model.summary()

# Visualize Neural Network Architecture
# Visualize the neural network layers using an external tool.
from Draw_ANN import drawANN
fig = drawANN([X_train.shape[1], 32, 64, y_label.shape[1]])

# Compile Model
# Compile the model with Adam optimizer and categorical crossentropy loss function.
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Train the model on the training data for 100 epochs with a batch size of 32.
# Training the model while tracking metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(100):  # Number of epochs
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    train_losses.append(history.history['loss'][0])
    val_losses.append(history.history['val_loss'][0])
    train_accuracies.append(history.history['accuracy'][0])
    val_accuracies.append(history.history['val_accuracy'][0])

# Plot Training vs Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# Plot Loss Distribution
plt.figure(figsize=(10, 6))
plt.hist(train_losses, bins=15, alpha=0.6, label='Training Loss Distribution', color='blue')
plt.hist(val_losses, bins=15, alpha=0.6, label='Validation Loss Distribution', color='orange')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.legend()
plt.title('Loss Distribution Over Epochs')
plt.show()


# Evaluate Model
# Evaluate the model performance on the test data.
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predictions and Evaluation
# Generate predictions and evaluate the model using a confusion matrix and classification report.
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.show()

# Classification Report
# Print the precision, recall, and F1-score for each class.
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))
