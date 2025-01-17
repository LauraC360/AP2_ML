import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

# Citirea datelor
def load_and_preprocess_data(file_path):
    tourism_data = pd.read_csv(file_path)

    # Preprocesarea datelor
    encoder = LabelEncoder()

    tourism_data['Category'] = encoder.fit_transform(
        tourism_data['Category'])  # Transformăm Category în variabilă numerică
    tourism_data['Accommodation_Available'] = encoder.fit_transform(
        tourism_data['Accommodation_Available'])  # Convertim 'Accommodation_Available'

    # Discretizarea venitului în 3 categorii
    bins = [0, 50000, 200000, float('inf')]  # Praguri pentru categorii
    labels = ['Low', 'Medium', 'High']
    tourism_data['Revenue_Category'] = pd.cut(tourism_data['Revenue'], bins=bins, labels=labels, right=False)

    # Selectarea caracteristicilor și țintei
    X = tourism_data[['Visitors', 'Rating', 'Accommodation_Available']]  # Caracteristici de intrare
    y = tourism_data['Revenue_Category']  # Variabila discretizată de predicție (Revenue_Category)

    # Împărțirea datelor în seturi de antrenament și testare (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder


# Antrenarea modelului ID3
def train_decision_tree(x_train, x_test, y_train, y_test, criterion='entropy'):
    # Crearea și antrenarea modelului ID3 (folosind entropia ca criteriu)
    model = DecisionTreeClassifier(criterion=criterion, random_state=42)
    model.fit(x_train, y_train)

    # Predicții
    predictions = model.predict(x_test)

    # Evaluare model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')

    return model, accuracy

# Salvarea rezultatelor evaluării într-un fișier CSV pentru comparație
def save_evaluation_results(results, output_file='evaluation_results.csv'):
    if os.path.exists(output_file):
        # Dacă fișierul există deja, citim datele existente și adăugăm noi rezultate
        existing_results = pd.read_csv(output_file)
        results_df = pd.DataFrame(results)
        updated_results = pd.concat([existing_results, results_df], ignore_index=True)
        updated_results.to_csv(output_file, index=False)
    else:
        # Dacă fișierul nu există, se creează unul nou cu noile rezultate
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    dataset_path = 'data/tourism_dataset.csv'
    X_train, X_test, Y_train, Y_test, data_encoder = load_and_preprocess_data(dataset_path)

    # Antrenare și evaluare ID3
    id3_model, id3_accuracy = train_decision_tree(X_train, X_test, Y_train, Y_test, criterion='entropy')

    # Salvarea rezultatelor într-un fișier CSV
    evaluation_results = [
        {'Algorithm': 'ID3', 'Accuracy': id3_accuracy}
    ]
    save_evaluation_results(evaluation_results)