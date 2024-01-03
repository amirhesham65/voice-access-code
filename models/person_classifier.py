import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle


class PersonClassifier:
    def __init__(self, data_path):
        data = pd.read_csv(data_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scalar = StandardScaler()
        self.X_train_scaled = self.scalar.fit_transform(self.X_train)
        self.X_test_scaled = self.scalar.transform(self.X_test)

        self.svm = SVC(kernel="rbf", C=1)

    def train(self) -> float:
        self.svm.fit(self.X_train_scaled, self.y_train)

        y_pred = self.svm.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def dump(self, pickle_path):
        pickle.dump(self.svm, open(pickle_path, "wb"))

    def predict(self, pickle_path, input_path):
        person_detection_loader = pickle.load(open(pickle_path, "rb"))
        X_new = pd.read_csv(input_path)
        X_new_scaled = self.scalar.transform(X_new)  # Use the same scaler from training
        y = person_detection_loader.predict(X_new_scaled)

        percentage_distribution = np.bincount(y) / len(y) * 100

        people = [
            "amir_hesham",
            "omar_emad",
            "farah_ossama",
            "omar_mohamed",
            "merna_abdelmoez",
            "mohamed_elsayed",
            "omar_nabil",
            "ossama_mohamed",
        ]
        result = {}
        for class_label, percentage in enumerate(percentage_distribution):
            result[people[class_label]] = percentage

        return result


if __name__ == "__main__":
    person_classifier = PersonClassifier(data_path="data/combined_train_speakers.csv")
    # person_classifier.train()
    # person_classifier.dump("person_detection_model.pkl")
    result = person_classifier.predict(
        pickle_path="pickles/person_detection_model.pkl",
        input_path="recorded_audio.csv",
    )
    print(result)
