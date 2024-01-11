import sys, wave
import threading
import matplotlib.pyplot as plt
from PyQt6 import QtWidgets, uic
import pyaudio
from scipy.io import wavfile

import test
from feature_extraction import extract_input_features
from test import compare_sentences
from models.person_classifier import PersonClassifier
from models.sentence_classifier import SentenceClassifier

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("mainwindow.ui", self)
        self.setWindowTitle("Voice Code Access")

        self.setup_classifiers()

        self.is_recording = False
        self.frames = []
        self.p = pyaudio.PyAudio()

        self.record_button.pressed.connect(self.start_stop_recording)
        self.reset_button.pressed.connect(self.reset)


    def reset(self):
        self.spectrogram_graph.canvas.axes.clear()
        self.spectrogram_graph.canvas.draw()
        person_label_map = {
            "amir_hesham": self.amir_hesham,
            "omar_emad": self.omar_emad,
            "farah_ossama": self.farah_ossama,
            "omar_mohamed": self.omar_mohamed,
            "merna_abdelmoez": self.merna_abdelmoez,
            "mohamed_elsayed": self.mohamed_elsayed,
            "omar_nabil": self.omar_nabil,
            "ossama_mohamed": self.ossama_mohamed,
        }
        for label in person_label_map.values():  
            label.setText('0%')
        
        sentence_label_map = {
            "open_middle_door": self.omd,
            "grant_me_access": self.gma,
            "unlock_the_gate": self.utg,
        }

        for label in sentence_label_map.values():
            label.setText('0%')    


    def setup_classifiers(self):
        data_path = "data/combined_train_speakers.csv"
        self.person_classifier = PersonClassifier(data_path=data_path)
        self.sentence_classifier = SentenceClassifier(data_path=data_path)

    def start_stop_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.setText("Stop")
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
        )
        self.frames = []
        threading.Thread(target=self._record).start()

    def stop_recording(self):
        self.is_recording = False
        self.record_button.setText("Record")
        self.stream.stop_stream()
        self.stream.close()
        self.save_recorded_audio()
        self.plot_spectrogram()
        extract_input_features("./input/recorded_audio.wav")

        # Get results
        persons_conf = self._predict_person()
        sentences_conf = self._predict_sentence()

        self._update_data(persons_conf, sentences_conf)

    def _record(self):
        while self.is_recording:
            data = self.stream.read(1024)
            self.frames.append(data)

    def save_recorded_audio(self):
        wf = wave.open("./input/recorded_audio.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(self.frames))
        wf.close()

    def plot_spectrogram(self):
        self.spectrogram_graph.canvas.axes.clear()
        sample_rate, signal = wavfile.read("./input/recorded_audio.wav")
        _, _, _, spectrogram = self.spectrogram_graph.canvas.axes.specgram(
            signal, Fs=sample_rate
        )
        plt.savefig("input_spectrogram.png")
        self.spectrogram_graph.canvas.draw()
        return spectrogram

    def _predict_person(self):
        result = self.person_classifier.predict(
            pickle_path="pickles/person_detection_model.pkl",
            input_path="./input/recorded_audio.csv",
        )
        return result

    def _predict_sentence(self):
        result = test.compare_sentences('./input/recorded_audio.wav')
        return result

    def _update_data(self, persons_conf, sent_conf):
        person_label_map = {
            "amir_hesham": self.amir_hesham,
            "omar_emad": self.omar_emad,
            "farah_ossama": self.farah_ossama,
            "omar_mohamed": self.omar_mohamed,
            "merna_abdelmoez": self.merna_abdelmoez,
            "mohamed_elsayed": self.mohamed_elsayed,
            "omar_nabil": self.omar_nabil,
            "ossama_mohamed": self.ossama_mohamed,
        }
        

        for person, label in person_label_map.items():
            if persons_conf.get(person):
                label.setText(str(int(persons_conf[person])) + "%")
            else:    
                label.setText('0%')

        # Update sentence prediction confidences
        sentence_label_map = {
            "open_middle_door": self.omd,
            "grant_me_access": self.gma,
            "unlock_the_gate": self.utg,
        }

        for sentence, label in sentence_label_map.items():
            label.setText(str(int(sent_conf[sentence])) + "%")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
