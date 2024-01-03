import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle


class SentenceClassifier:
    def __init__(self, data_path):
        # Read Data
        data = pd.read_csv(data_path)

        # Split Data
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale Features
        self.scalar = StandardScaler()
        self.X_train_scaled = self.scalar.fit_transform(self.X_train)
        self.X_test_scaled = self.scalar.transform(self.X_test)

    def train(self) -> float:
        # Train SVM Model with Best Parameters
        best_params = {"C": 10, "gamma": 0.1, "kernel": "rbf"}
        self.svm = SVC(**best_params)
        self.svm.fit(self.X_train_scaled, self.y_train)

        # Predict and Evaluate
        y_pred = self.svm.predict(self.X_test_scaled)

        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def dump(self, pickle_path):
        pickle.dump(self.svm, open(pickle_path, "wb"))

    def predict(self, pickle_path, input_path):
        sentence_detection_loader = pickle.load(open(pickle_path, "rb"))
        X_new = pd.read_csv(input_path)
        X_new_scaled = self.scalar.transform(X_new)  # Use the same scaler from training

        y_new_pred = sentence_detection_loader.predict(
            X_new_scaled
        )  # Use the trained model (best_svm)

        percentage_distribution = np.bincount(y_new_pred) / len(y_new_pred) * 100

        result = {
            "open_middle_door": percentage_distribution[8],
            "grant_me_access": percentage_distribution[9],
            "unlock_the_gate": percentage_distribution[10],
        }

        return result


# if __name__ == "__main__":
#     sentence_classifier = SentenceClassifier(data_path="combined_train_speakers.csv")
#     # sentence_classifier.train()
#     # sentence_classifier.dump("sentence_detection_model.pkl")
#     result = sentence_classifier.predict(
#         pickle_path="sentence_detection_model.pkl", input_path="utg.csv"
#     )
#     print(result)
