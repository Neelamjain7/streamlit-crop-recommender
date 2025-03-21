{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eff53ba4-56d4-4024-bd1e-7c3a42bff1b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# train_model.py\n",
    "\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import pickle\n",
    "\n",
    "# Load sample Crop Recommendation Dataset\n",
    "# You can download this from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n",
    "\n",
    "df = pd.read_csv(\"Crop_recommendation.csv\")  # Make sure this file is in the same folder\n",
    "\n",
    "# Features and Labels\n",
    "X = df.drop('label', axis=1)\n",
    "y = df['label']\n",
    "\n",
    "# Train-Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train Random Forest Classifier\n",
    "model = RandomForestClassifier()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Save the trained model to model.pkl\n",
    "with open(\"model.pkl\", \"wb\") as f:\n",
    "    pickle.dump(model, f)\n",
    "\n",
    "print(\"✅ Model trained and saved as model.pkl successfully!\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
