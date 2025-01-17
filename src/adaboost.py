import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import os

# Citirea și preprocesarea datelor
def load_and_preprocess_data(file_path, is_classification=True):
    data = pd.read_csv(file_path)

    # Preprocesarea datelor
    encoder = LabelEncoder()
    data['Category'] = encoder.fit_transform(data['Category'])
    data['Accommodation_Available'] = encoder.fit_transform(data['Accommodation_Available'])

    # Discretizarea venitului în 3 categorii
    if is_classification:
        bins = [0, 50000, 200000, float('inf')]
        labels = ['Low', 'Medium', 'High']
        data['Revenue_Category'] = pd.cut(data['Revenue'], bins=bins, labels=labels, right=False)
        y = data['Revenue_Category']
    else:
        y = data['Revenue']

    # Selectarea caracteristicilor (features) și a țintei (target)
    X = data[['Visitors', 'Rating', 'Accommodation_Available']]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, data


# Antrenarea modelului AdaBoost
def train_adaboost(x_train, x_test, y_train, y_test, is_classification=True, n_estimators=50):
    if is_classification:
        model = AdaBoostClassifier(n_estimators=n_estimators, algorithm='SAMME', random_state=42)
    else:
        model = AdaBoostRegressor(n_estimators=n_estimators, random_state=42)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # Evaluare model cu metricile Model Accuracy, Mean Squared Error și R2 Score
    if is_classification:
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy}')
        return model, accuracy
    else:
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R2 Score: {r2}')
        return model, mse, r2


# Vizualizarea predicțiilor
def visualize_predictions(y_test, predictions, is_classification=True, file_name='adaboost_predictions.png'):
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


# Salvarea rezultatelor evaluării într-un fișier CSV existent
def save_evaluation_results(results, output_file='evaluation_results.csv'):
    if os.path.exists(output_file):
        # Dacă fișierul există deja, se adaugă datele existente și se adaugă noile rezultate
        existing_results = pd.read_csv(output_file)
        results_df = pd.DataFrame(results)
        updated_results = pd.concat([existing_results, results_df], ignore_index=True)
        updated_results.to_csv(output_file, index=False)
    else:
        # Dacă fișierul nu există - creara unuia nou cu rezultatele actuale
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    dataset_path = 'data/tourism_dataset.csv'

    # Clasificare sau regresie
    classification = True

    X_train, X_test, Y_train, Y_test, tourism_data = load_and_preprocess_data(dataset_path, classification)

    # Parametri AdaBoost
    n_estimators = 100  # Numărul de estimatori
    if classification:
        adaboost_model, accuracy = train_adaboost(X_train, X_test, Y_train, Y_test, classification, n_estimators)
        evaluation_results = [{'Algorithm': 'AdaBoost', 'Accuracy': accuracy}]
    else:
        adaboost_model, mse, r2 = train_adaboost(X_train, X_test, Y_train, Y_test, classification=False, n_estimators=n_estimators)
        evaluation_results = [{'Algorithm': 'AdaBoost', 'Mean Squared Error': mse, 'R2 Score': r2}]

    # Salvarea rezultatelor într-un fișier CSV
    save_evaluation_results(evaluation_results)

    # Vizualizarea predicțiilor
    visualize_predictions(Y_test, adaboost_model.predict(X_test), classification, file_name='adaboost_predictions.png')