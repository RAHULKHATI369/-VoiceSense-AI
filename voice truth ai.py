import os
import torch
import pyttsx3
import speech_recognition as sr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

# Suppress unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

class VoiceSenseAI:
    def __init__(self):
        """Set up VoiceSense: voice, ears, brain."""
        self.voice = pyttsx3.init()
        self.voice.setProperty('rate', 155)
        self.ears = sr.Recognizer()

        # Load AI brains
        self.emotion_tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

        self.truth_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.truth_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")

        # Known users
        self.known_faces = {"rahul khati": {"title": "boss"}}
        self.current_user = None

    def talk(self, text):
        """VoiceSense speaks naturally."""
        print(f"🗣️ VoiceSense: {text}")
        self.voice.say(text)
        self.voice.runAndWait()
        time.sleep(0.5)

    def hear(self):
        """VoiceSense listens carefully."""
        try:
            with sr.Microphone() as mic:
                print("👂 Listening...")
                self.ears.adjust_for_ambient_noise(mic, duration=1)
                audio = self.ears.listen(mic, timeout=10)
                text = self.ears.recognize_google(audio).lower()
                print(f"👤 You: {text}")
                return text
        except Exception as e:
            self.talk("Oops, something went wrong while listening.")
            print(f"Audio error: {e}")
            return ""

    def feel_emotion(self, text):
        """VoiceSense reads emotions."""
        inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.emotion_model(**inputs)
        emotions = ["positive", "neutral", "negative"]
        probs = torch.softmax(outputs.logits, dim=1)
        return emotions[torch.argmax(probs)]

    def sense_truth(self, text):
        """VoiceSense detects truth or lie."""
        inputs = self.truth_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.truth_model(**inputs)
        labels = ["truth", "lie"]
        probs = torch.softmax(outputs.logits, dim=1)
        return labels[torch.argmax(probs)]

    def greet_user(self):
        """VoiceSense greets and recognizes."""
        self.talk("Hey there! Who am I talking to?")
        answer = self.hear()

        for name, info in self.known_faces.items():
            if name in answer:
                self.current_user = name
                self.talk(f"Welcome back {info['title']} {name.split()[0]}! Tell me, what are we feeling today?")
                return True

        self.talk("Hmm... I don't know you yet, but it's lovely to meet you!")
        return False

    def follow_commands(self):
        """VoiceSense follows user's emotional needs."""
        while True:
            command = self.hear()

            if not command:
                continue

            if "check his feelings" in command:
                self.talk("Alright, let me check how he is feeling. Please tell me what he said.")
                statement = self.hear()
                if statement:
                    mood = self.feel_emotion(statement)
                    emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                    self.talk(f"From what he said, he feels {mood} {emoji[mood]}.")

            elif "check her feelings" in command:
                self.talk("Sure, let me sense her feelings. Please share what she said.")
                statement = self.hear()
                if statement:
                    mood = self.feel_emotion(statement)
                    emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
                    self.talk(f"From what she said, she feels {mood} {emoji[mood]}.")

            elif "detect if he says truth or lie" in command:
                self.talk("Alright, let's check if he is being honest. What did he say?")
                statement = self.hear()
                if statement:
                    truth = self.sense_truth(statement)
                    self.talk(f"Listening carefully... I'd say he is telling the {truth}.")

            elif "detect if she says truth or lie" in command:
                self.talk("Got it, let's find out if she is being truthful. What did she say?")
                statement = self.hear()
                if statement:
                    truth = self.sense_truth(statement)
                    self.talk(f"Based on her words... she is telling the {truth}.")

            elif "thank you" in command or "bye" in command:
                self.talk("You're most welcome! I'll always be here when you need me. Bye!")
                break

            else:
                self.talk("Hmm, could you say it a little differently? I want to get it just right.")

    def start(self):
        """VoiceSense is ready."""
        self.talk("✨ Hello! VoiceSense AI is awake and ready to understand you.")
        if self.greet_user():
            self.follow_commands()

if __name__ == "__main__":
    try:
        ai = VoiceSenseAI()
        ai.start()
    except Exception as error:
        print(f"⚡ Error occurred: {error}")
    finally:
        print("🛑 VoiceSense has powered down.")
