import hupper
import chatbot
from dotenv import load_dotenv, find_dotenv

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    reloader = hupper.start_reloader('chatbot.main')
    chatbot.main()
