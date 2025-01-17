import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

# Citirea și preprocesarea datelor
def preprocess_data(file_path, selected_country):
    data = pd.read_csv(file_path)

    # Filtrarea doar pentru țara selectată
    data = data[data['Country'] == selected_country]

    # Calcularea profitului per vizitator
    data['Profit_Per_Visitor'] = data['Revenue'] / data['Visitors']

    # Crearea de etichete pentru categorii de activități bazate pe profit
    category_means = data.groupby('Category')['Profit_Per_Visitor'].mean().sort_values(ascending=False)
    ranking = {category: rank for rank, category in enumerate(category_means.index)}
    data['Category_Rank'] = data['Category'].map(ranking)

    # Codificarea coloanei 'Accommodation_Available' pentru a înlocui Yes cu 1 și No cu 0
    data['Accommodation_Available'] = data['Accommodation_Available'].map({'Yes': 1, 'No': 0})

    # Caracteristici și țintă
    X = data[['Visitors', 'Rating', 'Accommodation_Available']]
    y = data['Category_Rank']

    # Standardizare și împărțire în seturi de antrenament și testare
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, category_means


# Antrenarea modelului AdaBoost
def train_adaboost(X_train, X_test, y_train, y_test, n_estimators=50):
    # Crearea modelului AdaBoost
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Predicții
    y_pred = model.predict(X_test)

    # Evaluare
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(len(set(y_test)))])

    print(f"Accuracy: {accuracy}")
    print(report)

    return model, accuracy, report

# Salvarea rezultatelor în fișierul CSV
def save_evaluation_results(results, output_file='evaluation_results.csv'):
    if os.path.exists(output_file):
        # Dacă fișierul există deja, citim datele existente și adăugăm noi rezultate
        existing_results = pd.read_csv(output_file)
        results_df = pd.DataFrame(results)
        updated_results = pd.concat([existing_results, results_df], ignore_index=True)
        updated_results.to_csv(output_file, index=False)
    else:
        # Dacă fișierul nu există - crearea unului nou cu noile rezultate
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)


# Salvarea rezultatelor finale într-un fișier CSV și TXT
def save_results(category_means, accuracy, report, output_file_csv, output_file_txt):
    # Salvare ranking într-un fișier CSV
    category_means.to_csv(output_file_csv, header=['Profit_Per_Visitor'])

    # Salvare metrici într-un fișier TXT
    with open(output_file_txt, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(f"Classification Report:\n{report}\n")


if __name__ == "__main__":
    dataset_path = 'data/tourism_dataset.csv'  # Calea către fișierul CSV
    selected_country = 'USA'  # Am luat în considerare doar datele pentru această țară, USA
    output_csv = 'category_ranking.csv'
    output_txt = 'adaboost_results.txt'
    evaluation_results_csv = 'evaluation_results.csv'

    # Preprocesare
    X_train, X_test, y_train, y_test, category_means = preprocess_data(dataset_path, selected_country)

    # Antrenare AdaBoost
    adaboost_model, accuracy, report = train_adaboost(X_train, X_test, y_train, y_test)

    # Salvare rezultate
    save_results(category_means, accuracy, report, output_csv, output_txt)

    # Adăugare rezultate AdaBoost în fișierul CSV de evaluare
    evaluation_results = [{'Algorithm': 'AdaBoost optimized', 'Accuracy': accuracy}]
    save_evaluation_results(evaluation_results, evaluation_results_csv)