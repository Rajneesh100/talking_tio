import random
import time
import threading
from llama_generate import LlamaSession, speak_text
from voice_input import continuous_voice_listener
import requests


MEMORY_URL = "http://localhost:8000"
session = LlamaSession()

# Critical section lock
interaction_lock = threading.Lock()
last_voice_interaction_time = 0
VOICE_COOLDOWN = 300  # 5 minutes in seconds

def store_memory(text, source="agent"):
    try:
        requests.post(f"{MEMORY_URL}/memory/add", json={"text": text, "source": source}, timeout=3)
    except Exception as e:
        print("memory store failed:", e)

def query_memory(q, k=3):
    try:
        r = requests.post(f"{MEMORY_URL}/memory/query", json={"q": q, "k": k}, timeout=3).json()
        return r.get("results", [])
    except Exception as e:
        print("memory query failed:", e)
        return []

def choose_emotion_and_prompt(topic=None):
    emotions = ["neutral", "happy", "amused", "nostalgic", "sad"]
    emotion = random.choice(emotions)
    if not topic:
        topic = random.choice(["life", "coding", "music", "food"])
    # have a llm genrate this prompt dynamically
    prompt = f"You are a warm funny friend. Keep it short, witty and kind. Topic: {topic}. Use a tone: {emotion}. Speak as a human would in 1-3 sentences. and ask some thing to Rajneesh, just come up with some random shit"
    return emotion, prompt

def spaontenuius_loop():
    print("Starting Tío loop. ctrl+c to stop.")
    print("Type your response or press Enter to skip...\n")
    while True:
        print("here ")
        wait = random.randint(30,40)
        time.sleep(wait)
        emotion, prompt = choose_emotion_and_prompt()
        try:
            print(f"spontenious [{emotion}]:", prompt)
            intraction(prompt,source="spontaneous")
        except Exception as e:
            print("error:", e)
            text = f"(error generating) i wanted to say something about {prompt}"

        
        # small chance to query memory and reflect
        if random.random() < 0.1:
            mems = query_memory("recall some fun chats", k=1)
            if mems:
                reflection = f"Remember when we said: {mems[0]['text'][:120]}..."
                print("Reflection:", reflection)
                speak_text(reflection)



# critical section source can be either voice_input or spontenious input
def intraction(prompt, source):
    global last_voice_interaction_time
    
    # Check if this is a spontaneous call and voice was used recently
    if source == "spontaneous":
        time_since_voice = time.time() - last_voice_interaction_time
        if time_since_voice < VOICE_COOLDOWN:
            remaining = int(VOICE_COOLDOWN - time_since_voice)
            print(f"🔇 Main loop paused ({remaining}s remaining after voice input)")
            return
    
    # Try to acquire lock (non-blocking for spontaneous, blocking for voice)
    if source == "spontaneous":
        acquired = interaction_lock.acquire(blocking=False)
        if not acquired:
            print("🔒 Interaction in progress, skipping spontaneous message")
            return
    else:
        # Voice input blocks until it gets the lock
        print("🎤 Voice input waiting for lock...")
        interaction_lock.acquire()
        last_voice_interaction_time = time.time()
    
    try:
        print(f"🔓 Lock acquired by: {source}")
        store_memory(prompt, source)
        response = session.chat(prompt)
        print(f"Tío: {response}")
        store_memory(response, source="llm")
        speak_text(response)
    except Exception as e:
        print(f"❌ Error in interaction: {e}")
    finally:
        interaction_lock.release()
        print(f"🔓 Lock released by: {source}")

    


if __name__ == "__main__":
    print("🚀 Starting Tío Agent with Voice Input...")
    print("=" * 50)
    
    # Start voice listener in a separate thread (pass intraction as callback)
    voice_thread = threading.Thread(target=lambda: continuous_voice_listener(intraction), daemon=True)
    voice_thread.start()
    print("✅ Voice listener thread started")
    
    # Start spontaneous loop in a separate thread
    spontaneous_thread = threading.Thread(target=spaontenuius_loop, daemon=True)
    spontaneous_thread.start()
    print("✅ Spontaneous loop thread started")
    
    print("=" * 50)
    print("🎙️  Both systems running in parallel!")
    print("Press Ctrl+C to stop...\n")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        print("Goodbye!")
