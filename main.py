from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import sqlite3
from data import data_entries
import speech_recognition as sr

conn = sqlite3.connect('text_entries.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
''')
conn.commit()

data = data_entries

texts, labels = zip(*data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

clf = MultinomialNB()
clf.fit(X, labels)

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening... (say 'stop listening' to switch back to typing)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected. Try again.")
            return ""

    try:
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text
    except sr.UnknownValueError:
        print("🤖 Sorry, I couldn't understand that.")
        return ""
    except sr.RequestError:
        print("❌ Could not request results from the speech service.")
        return ""

def classify_input(text):
    X_test = vectorizer.transform([text])
    prediction = clf.predict(X_test)[0]
    return prediction  # 0 = Statement, 1 = Query

def save_statement(content):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('INSERT INTO entries (content, timestamp) VALUES (?, ?)', (content, timestamp))
    conn.commit()

def get_entries():
    cursor.execute('SELECT * FROM entries')
    return cursor.fetchall()

def respond_to_query(text):
    history = get_entries()
    context = f'''{history}'''

    template = """
Answer the question below.
If I ever said something like next week or next month, calculate the date for it from the timestamp aswell. 
Don't mention the previous timestamp and just tell me the date.
Below is the diary of a person and that same person is asking you questions.

Context: {context}

Questions: {question}

Answer:
"""
    model = OllamaLLM(model = "gemma3")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model


    result = chain.invoke({"context": context, "question": text})
    return result

def delete_all_entries():
    cursor.execute('DELETE FROM entries')
    conn.commit()
    print("🗑️  All entries have been deleted.")

def show_entries_with_ids():
    entries = get_entries()
    if not entries:
        print("📂 No entries found.")
    else:
        print("📜 Saved Entries:")
        for entry in entries:
            print(f"{entry[0]}. [{entry[2]}] {entry[1]}")

def delete_entry(entry_id):
    cursor.execute('DELETE FROM entries WHERE id = ?', (entry_id,))
    conn.commit()
    print(f"🗑️  Entry #{entry_id} deleted.")


if __name__ == "__main__":
    print("""

███╗░░░███╗███████╗███╗░░░███╗░█████╗░██████╗░███████╗░█████╗░░██████╗███████╗
████╗░████║██╔════╝████╗░████║██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝
██╔████╔██║█████╗░░██╔████╔██║██║░░██║██████╔╝█████╗░░███████║╚█████╗░█████╗░░
██║╚██╔╝██║██╔══╝░░██║╚██╔╝██║██║░░██║██╔══██╗██╔══╝░░██╔══██║░╚═══██╗██╔══╝░░
██║░╚═╝░██║███████╗██║░╚═╝░██║╚█████╔╝██║░░██║███████╗██║░░██║██████╔╝███████╗
╚═╝░░░░░╚═╝╚══════╝╚═╝░░░░░╚═╝░╚════╝░╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝╚═════╝░╚══════╝

          
Functions:
    1. ❌ to quit type 'bye MemorEase'
    2. 🗑️  to delete all entries type 'delete all'
    3. 🗑️  to delete one entry type 'delete one'
    4. 📜 to show all entries type 'history'
""")
    while True:
        mode = input("Type 'voice' to speak or press Enter to type: ").strip().lower()

        if mode == "voice":
            inp = get_voice_input()
            if inp.lower() == "stop listening":
                continue
        else:
            inp = mode

        if not inp.strip():
            continue

        if inp.lower() == "bye memorease":
            print("Thank you for using MemorEase :).")
            break
        elif inp.lower() == "history":
            entries = get_entries()
            if not entries:
                print("📂 No entries found.")
                print()
            else:
                print("📜 Saved Entries:")
                for entry in entries:
                    print(f"[{entry[2]}] {entry[1]}")
                print()
            continue
        elif inp.lower() == "delete all":
            entries = get_entries()
            if not entries:
                print("📂 No entries found.")
                print()
            else:
                delete_all_entries()
                print()
            continue
        elif inp.lower() == "delete one":
            entries = get_entries()
            if not entries:
                print("📂 No entries found.")
                print()
            else:
                show_entries_with_ids()
                id_delete = input("Enter id to be deleted: ")
                if id_delete.isnumeric():
                    id_delete = int(id_delete)
                    delete_entry(id_delete)
                    print()
                else:
                    print("Invalid ID")
                    print()
            continue

        classification = classify_input(inp)
        if classification == 0:
            save_statement(inp)
            print("📝 Statement saved.\n")
        elif classification == 1:
            response = respond_to_query(inp)
            print("AI:", response)
