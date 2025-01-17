import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
import numpy as np


# Citirea și preprocesarea datelor
def load_and_preprocess_data(file_path):
    tourism_data = pd.read_csv(file_path)

    # Preprocesarea datelor
    encoder = LabelEncoder()
    tourism_data['Category'] = encoder.fit_transform(tourism_data['Category'])
    tourism_data['Accommodation_Available'] = encoder.fit_transform(tourism_data['Accommodation_Available'])

    # Discretizarea venitului în 3 categorii
    bins = [0, 50000, 200000, float('inf')]
    labels = ['Low', 'Medium', 'High']
    tourism_data['Revenue_Category'] = pd.cut(tourism_data['Revenue'], bins=bins, labels=labels, right=False)

    # Selectarea caracteristicilor și țintei
    X = tourism_data[['Visitors', 'Rating', 'Accommodation_Available']]
    y = tourism_data['Revenue_Category']

    # Împărțirea datelor în seturi de antrenament și testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Model ID3
def train_id3(x_train, x_test, y_train, y_test, criterion='entropy'):
    model = DecisionTreeClassifier(criterion=criterion, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return predictions


# Model kNN
def train_knn(X_train, X_test, y_train, y_test, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


# Model AdaBoost
def train_adaboost(x_train, x_test, y_train, y_test, n_estimators=50):
    model = AdaBoostClassifier(n_estimators=n_estimators, algorithm='SAMME', random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return predictions


# Calcul MSE
def calculate_mse(y_test, predictions_dict):
    mse_dict = {}
    for algorithm, predictions in predictions_dict.items():
        mse = mean_squared_error(
            np.array(y_test.cat.codes),  # Convertim valorile reale la coduri numerice pentru MSE
            np.array(pd.Categorical(predictions, categories=y_test.cat.categories).codes)  # Coduri predicții
        )
        mse_dict[algorithm] = mse
    return mse_dict


# Grafic pie chart
def plot_mse_pie_chart(mse_dict, file_name='mse_pie_chart.png'):
    plt.figure(figsize=(8, 8))
    plt.pie(mse_dict.values(), labels=mse_dict.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Proporția MSE pentru fiecare algoritm', fontsize=14)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    dataset_path = 'data/tourism_dataset.csv'

    # Preprocesarea datelor
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data(dataset_path)

    # Antrenarea algoritmilor și obținerea predicțiilor
    id3_predictions = train_id3(X_train, X_test, Y_train, Y_test)
    knn_predictions = train_knn(X_train, X_test, Y_train, Y_test, k=5)
    adaboost_predictions = train_adaboost(X_train, X_test, Y_train, Y_test)

    # Crearea dicționarului cu predicții
    predictions_dict = {
        'ID3': id3_predictions,
        'kNN': knn_predictions,
        'AdaBoost': adaboost_predictions
    }

    # Calcularea MSE pentru fiecare algoritm
    mse_dict = calculate_mse(Y_test, predictions_dict)

    # Generarea pie chart
    plot_mse_pie_chart(mse_dict)