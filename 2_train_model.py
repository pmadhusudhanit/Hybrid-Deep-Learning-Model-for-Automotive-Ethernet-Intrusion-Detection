import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tqdm import tqdm  # For progress tracking

# Step 1: Load preprocessed data
print("\nStep 1: Loading preprocessed data...")
df = pd.read_csv('preprocessed2_data.csv')
print("Loaded data summary:")
print(df.info())
print(df.head())

# Step 2: Separate features and labels
print("\nStep 2: Separating features and labels...")
X = df.drop(columns=['label'])
y = df['label']
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Step 3: Feature selection using Salp Swarm Algorithm (SSA)
def salp_swarm_algorithm(X, y, num_features=10):
    print("\nStep 3: Performing feature selection using SSA...")
    # Placeholder for SSA implementation
    # For now, select the top `num_features` based on correlation with the label
    correlations = X.corrwith(y).abs()
    selected_features = correlations.nlargest(num_features).index
    print(f"Selected features: {list(selected_features)}")
    return X[selected_features]

X_selected = salp_swarm_algorithm(X, y, num_features=10)
print("Feature selection complete. Selected features:")
print(X_selected.head())

# Save selected features to a CSV
selected_features_file = 'selected_features.csv'
pd.concat([X_selected, y], axis=1).to_csv(selected_features_file, index=False)
print(f"Selected features and labels saved to '{selected_features_file}'.")

# Step 4: Split data into training and testing sets
print("\nStep 4: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Save training and testing data to CSV
train_file = 'train_data.csv'
test_file = 'test_data.csv'
pd.concat([X_train, y_train], axis=1).to_csv(train_file, index=False)
pd.concat([X_test, y_test], axis=1).to_csv(test_file, index=False)
print(f"Training data saved to '{train_file}'.")
print(f"Testing data saved to '{test_file}'.")

# Step 5: Build and train DenseNet
def build_densenet(input_shape):
    print("\nStep 5: Building the DenseNet model...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("DenseNet model built successfully.")
    return model

print("Building and training the model...")
model = build_densenet(X_train.shape[1])

print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1  # Show progress bars for each epoch
)

# Save the trained model
model_file = 'intrusion_detection_model.h5'
model.save(model_file)
print(f"\nModel training complete. Model saved to '{model_file}'.")

# Step 6: Evaluate the model on the test set
print("\nStep 6: Evaluating the model on the test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
