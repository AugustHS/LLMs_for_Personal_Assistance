import sys
import threading
import os

from emotion_detection.main import emotion_detection
from chat_app import start_chat_app

if __name__ == '__main__':
    emotion_thread = threading.Thread(target=emotion_detection)
    emotion_thread.daemon = True
    emotion_thread.start()

    start_chat_app()
