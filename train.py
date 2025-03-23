import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load sample Crop Recommendation Dataset
# You can download this from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

df = pd.read_csv("Crop_recommendation.csv")  # Make sure this file is in the same folder

# Features and Labels
X = df.drop('label', axis=1)
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl successfully!")