import sqlite3
import re
import html
import bcrypt
import urllib.parse

# 1. Vulnerable: Directory Traversal vulnerability
def view_file_vulnerable(path):
    try:
        with open(path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        return "File not found"

# 2. Vulnerable: SQL Injection vulnerability
def search_user_vulnerable(query):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    # Vulnerable to SQL Injection
    cursor.execute(f"SELECT * FROM users WHERE username='{query}'")
    result = cursor.fetchall()
    connection.close()
    return result

# 3. Non-vulnerable: Using parameterized queries to prevent SQL Injection
def search_user_safe(query):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    # Safe from SQL Injection using parameterized query
    cursor.execute("SELECT * FROM users WHERE username=?", (query,))
    result = cursor.fetchall()
    connection.close()
    return result

# 4. Non-vulnerable: Validating phone numbers with regex
def process_phone_number_safe(phone_number):
    # Only allow 10-digit numbers (no special characters or spaces)
    if re.match(r'^\d{10}$', phone_number):
        return f"Phone number {phone_number} is valid."
    else:
        return "Invalid phone number."


# Example usage
if __name__ == "__main__":
    username = input("Enter username: ")
    phone_number = input("Enter phone number: ")
    password = input("Enter password: ")
    comment = input("Enter your comment: ")

    print("\nResults from functions:")

    # Vulnerable function (directory traversal)
    path = input("Enter file path to view (Vulnerable to directory traversal): ")
    print("Vulnerable File Viewing Function:")
    print(view_file_vulnerable(path))

    # Vulnerable function (SQL Injection)
    print("Vulnerable SQL Injection Function:")
    print(search_user_vulnerable(username))

    # Safe function (SQL injection prevention)
    print("Safe SQL Function (parameterized queries):")
    print(search_user_safe(username))

    # Safe phone number validation
    print("Safe Phone Number Function:")
    print(process_phone_number_safe(phone_number))
