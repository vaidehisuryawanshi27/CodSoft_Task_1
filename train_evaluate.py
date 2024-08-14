import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Define file paths
train_data_path = r'C:\Users\Vaidehi Suryawanshi\Downloads\movie genre classification\Genre Classification Dataset\train.csv'
test_data_path = r'C:\Users\Vaidehi Suryawanshi\Downloads\movie genre classification\Genre Classification Dataset\test.csv'
test_solution_path = r'C:\Users\Vaidehi Suryawanshi\Downloads\movie genre classification\Genre Classification Dataset\test_data_solution.csv'
model_path = r'logistic_regression_model.pkl'
vectorizer_path = r'tfidf_vectorizer.pkl'
label_encoder_path = r'label_encoder.pkl'

def load_data(file_path):
    """Load and parse data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Combine TITLE and DESCRIPTION into a single feature."""
    data['text'] = data['TITLE'] + ' ' + data.get('DESCRIPTION', '')
    return data

def train_model(X_train, y_train):
    """Train the Logistic Regression model."""
    vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.7, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    model = LogisticRegression(max_iter=100, multi_class='auto', solver='liblinear')
    model.fit(X_train_vectorized, y_train_encoded)
    
    return model, vectorizer, label_encoder

def save_model(model, vectorizer, label_encoder):
    """Save the model, vectorizer, and label encoder to disk."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)

def load_model():
    """Load the model, vectorizer, and label encoder from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, vectorizer, label_encoder

def evaluate_model(model, vectorizer, label_encoder, X_test, y_true):
    """Evaluate the model on test data and print predictions and accuracy."""
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    results_df = pd.DataFrame({
        'True_GENRE': y_true,
        'Predicted_GENRE': y_pred_labels
    })
    
    print("True Genre\tPredicted Genre")
    print("-" * 40)
    for _, row in results_df.iterrows():
        print(f"{row['True_GENRE']}\t{row['Predicted_GENRE']}")
    
    accuracy = (results_df['True_GENRE'] == results_df['Predicted_GENRE']).mean()
    print(f"\nModel Accuracy: {accuracy:.2f}")

def main():
    # Load and preprocess training data
    train_data = load_data(train_data_path)
    train_data = preprocess_data(train_data)
    
    # Train model
    model, vectorizer, label_encoder = train_model(train_data['text'], train_data['GENRE'])
    
    # Save the trained model
    save_model(model, vectorizer, label_encoder)
    
    # Load and preprocess test data and test solution
    test_data = load_data(test_data_path)
    test_data = preprocess_data(test_data)
    test_solution_data = load_data(test_solution_path)
    
    # Load model, vectorizer, and label encoder
    model, vectorizer, label_encoder = load_model()
    
    # Evaluate model
    evaluate_model(model, vectorizer, label_encoder, test_data['text'], test_solution_data['GENRE'])

if __name__ == "__main__":
    main()
