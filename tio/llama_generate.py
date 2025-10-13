# llama_session.py
import requests



def speak_text(text):
    try:
        requests.post(
            'http://localhost:1098/speak',
            json={"text": text},
        )
    except Exception as e:
        print(f"TTS Error: {e}")

class LlamaSession:
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.session_id = None  # generated automatically by ollama
        self.history = []

    def chat(self, user_msg):
        self.history.append({"role": "user", "content": user_msg})
        body = {
            "model": self.model,
            "messages": self.history,
            "stream": False,
            # you can optionally keep it alive longer
            "keep_alive": "24h"
        }
        r = requests.post(f"{self.base_url}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        # extract the model's message
        reply = data["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply


if __name__ == "__main__":
    # Test code - only runs when script is executed directly
    session = LlamaSession()
    print(session.chat("i'm Rajneesh who are you"))
    print("--------------")
    print(session.chat("remember my name?"))
