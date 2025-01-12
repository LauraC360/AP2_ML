import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_analyze_data(file_path):
    try:
        tourism_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
        return
    except pd.errors.ParserError:
        print("Error parsing the file.")
        return

    print("Preview of the data:")
    print(tourism_data.head())
    print("\nData Info:")
    print(tourism_data.info())

    # Check if 'Category' column exists
    if 'Category' in tourism_data.columns:
        # Visualize distribution of categories
        sns.countplot(tourism_data['Category'])
        plt.title("Distribution of Categories")
        plt.show()
    else:
        print("'Category' column not found in the data.")

    # Visualize correlations between variables
    sns.heatmap(tourism_data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    return tourism_data

if __name__ == "__main__":
    file_path = "data/tourism_dataset.csv"
    data = load_and_analyze_data(file_path)