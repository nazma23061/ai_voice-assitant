
import sqlite3

def create_database():

    connection = sqlite3.connect("conversations.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_message(role, message):

    connection = sqlite3.connect("conversations.db")

    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO conversations (role, message) VALUES (?, ?)",
        (role, message)
    )

    connection.commit()
    connection.close()


def get_messages():

    connection = sqlite3.connect("conversations.db")

    cursor = connection.cursor()

    cursor.execute(
        "SELECT role, message FROM conversations ORDER BY id"
    )

    messages = cursor.fetchall()

    connection.close()

    return [
        {"role": role, "content": message}
        for role, message in messages
    ]


def clear_messages():

    connection = sqlite3.connect("conversations.db")

    cursor = connection.cursor()

    cursor.execute("DELETE FROM conversations")

    connection.commit()
    connection.close()
