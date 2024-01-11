import speech_recognition as sr
from fuzzywuzzy import fuzz

open_middle_door_sentence = "samples/omar/open middle door/omar_omd1.wav"
grant_me_access_sentence = "samples/omar/grant me access/omar_gma1.wav"
unlock_the_gate_sentence = "samples/omar/unlock the gate/omar_utg1.wav"

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

    sim_score_with_omd = fuzz.ratio(test_sample, omd_transcription)
    sim_score_with_gma = fuzz.ratio(test_sample, gma_transcription)
    sim_score_with_utg = fuzz.ratio(test_sample, utg_transcription)

    sentence_similarity_dict = {'open_middle_door': sim_score_with_omd,
                                'grant_me_access': sim_score_with_gma,
                                'unlock_the_gate': sim_score_with_utg}
    print(sentence_similarity_dict)

    # Normalize scores
    def normalize_scores(scores_dict):
        total_score = sum(scores_dict.values())

        if total_score == 0:
            print("Total score is zero, cannot normalize.")
            return scores_dict

        normalized_dict = {key: round(((value / total_score) * 100), 3) for key, value in scores_dict.items()}
        return normalized_dict

    normalized_similarity_dict = normalize_scores(sentence_similarity_dict)
    print("Normalized Scores:", normalized_similarity_dict)

    return normalized_similarity_dict
