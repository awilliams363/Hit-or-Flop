# classification.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np



def load_and_merge_data():
    movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)
    credits = pd.read_csv("data/credits.txt", low_memory=False)

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    credits["id"] = pd.to_numeric(credits["id"], errors="coerce")

    merged = movies.merge(credits, on="id", how="inner")
    print("Merged dataset size:", merged.shape)

    return merged

def preprocess(df):
    # Remove rows missing important values
    df = df.dropna(subset=["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"])

    # Ensure numeric formatting on key columns
    df[["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]] = (
        df[["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]]
        .apply(pd.to_numeric, errors="coerce")
    )

    # Create profit and hit label
    df["profit"] = df["revenue"] - df["budget"]
    df["hit"] = (df["profit"] > 0).astype(int)

    # Selected numerical features
    features = ["budget", "runtime", "popularity", "vote_average", "vote_count"]
    df = df.dropna(subset=features)

    X = df[features]
    y = df["hit"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---- kNN Model ----
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    print("----- KNN Results -----")
    print("Accuracy:", accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))

    # ---- Naive Bayes Model ----
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    print("----- Naive Bayes Results -----")
    print("Accuracy:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb))

        # ---- CONFUSION MATRIX VISUALIZATION ----
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
    plt.title("kNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens")
    plt.title("Naive Bayes Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ---- RMSE & MAE FOR KNN ----
    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    mae_knn = mean_absolute_error(y_test, y_pred_knn)

    print("KNN RMSE:", rmse_knn)
    print("KNN MAE:", mae_knn)

    # ---- RMSE & MAE FOR NAIVE BAYES ----
    rmse_nb = np.sqrt(mean_squared_error(y_test, y_pred_nb))
    mae_nb = mean_absolute_error(y_test, y_pred_nb)

    print("Naive Bayes RMSE:", rmse_nb)
    print("Naive Bayes MAE:", mae_nb)


if __name__ == "__main__":
    df = load_and_merge_data()
    X, y = preprocess(df)
    train_models(X, y)

