import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Citirea datelor
def load_and_preprocess_data(file_path):
    tourism_data = pd.read_csv(file_path)

    # Preprocesarea datelor
    # Transformăm variabilele categorice în valori numerice
    encoder = LabelEncoder()

    tourism_data['Category'] = encoder.fit_transform(tourism_data['Category'])  # Transformăm Category în variabilă numerică
    tourism_data['Accommodation_Available'] = encoder.fit_transform(
        tourism_data['Accommodation_Available'])  # Convertim 'Accommodation_Available'

    # Discretizăm venitul în 3 categorii
    bins = [0, 50000, 200000, float('inf')]  # Praguri pentru categorii
    labels = ['Low', 'Medium', 'High']
    tourism_data['Revenue_Category'] = pd.cut(tourism_data['Revenue'], bins=bins, labels=labels, right=False)

    # Selectăm caracteristicile (features) și ținta (target)
    X = tourism_data[['Visitors', 'Rating', 'Accommodation_Available']]  # Caracteristici de intrare
    y = tourism_data['Revenue_Category']  # Variabila discretizată de predicție (Revenue_Category)

    # Împărțim datele în seturi de antrenament și testare (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder


# Antrenarea modelului
def train_decision_tree(x_train, x_test, y_train, y_test):
    # Creăm și antrenăm modelul ID3 (folosind entropia ca criteriu)
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(x_train, y_train)

    # Predicții
    predictions = model.predict(x_test)

    # Evaluare model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')

    return model


# Vizualizarea arborelui de decizie
def plot_decision_tree(dt_model, x, file_name="id3_tree.png"):
    # Vizualizarea arborelui de decizie folosind plot_tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, filled=True, feature_names=x.columns, class_names=dt_model.classes_, rounded=True, fontsize=12)
    plt.title('Arbore de decizie ID3')

    # Salvare imagine într-un fișier PNG
    plt.savefig(file_name, format='png')
    plt.close()  # Închidem figura pentru a elibera memoria


if __name__ == "__main__":
    dataset_path = 'data/tourism_dataset.csv'
    X_train, X_test, Y_train, Y_test, data_encoder = load_and_preprocess_data(dataset_path)
    model = train_decision_tree(X_train, X_test, Y_train, Y_test)
    plot_decision_tree(model, X_train, "id3_tree.png")