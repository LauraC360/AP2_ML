import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
import os

# Citirea și preprocesarea datelor
def load_and_preprocess_data(file_path, is_classification=True):
    tourism_data = pd.read_csv(file_path)

    # Preprocesarea datelor
    encoder = LabelEncoder()
    tourism_data['Category'] = encoder.fit_transform(tourism_data['Category'])
    tourism_data['Accommodation_Available'] = encoder.fit_transform(tourism_data['Accommodation_Available'])

    # Discretizarea venitului în 3 categorii
    if is_classification:
        bins = [0, 50000, 200000, float('inf')]
        labels = ['Low', 'Medium', 'High']
        tourism_data['Revenue_Category'] = pd.cut(tourism_data['Revenue'], bins=bins, labels=labels, right=False)
        y = tourism_data['Revenue_Category']
    else:
        y = tourism_data['Revenue']

    X = tourism_data[['Visitors', 'Rating', 'Accommodation_Available']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, tourism_data

# Antrenarea modelului kNN
def train_knn(X_train, X_test, y_train, y_test, k=5, is_classification=True):
    if is_classification:
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        model = KNeighborsRegressor(n_neighbors=k)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if is_classification:
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy}')
    else:
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R2 Score: {r2}')

    return model, predictions

# Vizualizarea predicțiilor
def visualize_predictions(y_test, predictions, is_classification=True, file_name='knn_predictions.png'):
    plt.figure(figsize=(12, 8))
    if is_classification:
        plt.scatter(range(len(y_test)), y_test, label='Valori reale', alpha=0.7, color='blue', marker='o')
        plt.scatter(range(len(predictions)), predictions, label='Predicții', alpha=0.7, color='orange', marker='x')
        plt.title('Compararea valorilor reale și a predicțiilor (Clasificare)', fontsize=16)
    else:
        plt.plot(y_test.reset_index(drop=True), label='Valori reale', alpha=0.7, color='blue')
        plt.plot(predictions, label='Predicții', alpha=0.7, color='orange')
        plt.title('Compararea valorilor reale și a predicțiilor (Regresie)', fontsize=16)

    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Valoare', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)

    # Exportarea graficului în PNG
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

# Salvarea rezultatelor evaluării într-un fișier CSV pentru comparație
def save_evaluation_results(results, output_file='evaluation_results.csv'):
    if os.path.exists(output_file):
        # Dacă fișierul există deja, citim datele existente și adăugăm noi rezultate
        existing_results = pd.read_csv(output_file)
        results_df = pd.DataFrame(results)
        updated_results = pd.concat([existing_results, results_df], ignore_index=True)
        updated_results.to_csv(output_file, index=False)
    else:
        # Dacă fișierul nu există, creăm unul nou cu noile rezultate
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    dataset_path = 'data/tourism_dataset.csv'

    # Clasificare sau regresie
    classification = True  # Schimbă în False pentru regresie

    X_train, X_test, Y_train, Y_test, tourism_data = load_and_preprocess_data(dataset_path, classification)
    k = 5  # Numărul de vecini
    knn_model, predictions = train_knn(X_train, X_test, Y_train, Y_test, k, classification)

    # Salvarea rezultatelor pentru kNN într-un fișier CSV
    evaluation_results = []
    if classification:
        accuracy = accuracy_score(Y_test, predictions)
        evaluation_results.append({'Algorithm': 'kNN Classification', 'Accuracy': accuracy})
    else:
        mse = mean_squared_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)
        evaluation_results.append({'Algorithm': 'kNN Regression', 'MSE': mse, 'R2 Score': r2})

    # Salvarea rezultatelor de evaluare într fișierul CSV
    save_evaluation_results(evaluation_results)

    # Vizualizarea predicțiilor
    visualize_predictions(Y_test, predictions, classification)