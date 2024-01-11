import speech_recognition as sr
from fuzzywuzzy import fuzz
import pickle

def transcribe_wav(file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def compare_sentences(test_sentence):
    omd_transcription = "open middle door"
    gma_transcription = "grant me access"
    utg_transcription = "unlock the gate"

    test_sample = transcribe_wav(test_sentence)

    sim_score_with_omd = min(fuzz.ratio(test_sample, omd_transcription), 90)
    sim_score_with_gma = min(fuzz.ratio(test_sample, gma_transcription), 90)
    sim_score_with_utg = min(fuzz.ratio(test_sample, utg_transcription), 90)

    sentence_similarity_dict = {'open_middle_door': sim_score_with_omd,
                                'grant_me_access': sim_score_with_gma,
                                'unlock_the_gate': sim_score_with_utg}
    # print(sentence_similarity_dict)

    return sentence_similarity_dict

# Save the function as a pickle file
with open('pickles/sentence_detection.pkl', 'wb') as file:
    pickle.dump(compare_sentences, file)