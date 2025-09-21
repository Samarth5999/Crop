import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model():
    # Load the dataset
    df = pd.read_csv('data/Crop_recommendation.csv')

    # Split the data into features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'crop_yield_dashboard/crop_model.pkl')

if __name__ == '__main__':
    train_model()
