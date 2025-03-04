import sqlite3
import re
import bcrypt
import urllib.parse
import requests
import os
import secrets
import logging

# 1. Vulnerable: Insecure Direct Object Reference (IDOR)
def get_user_data_vulnerable(user_id):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id={user_id}")  # No authorization check!
    result = cursor.fetchone()
    connection.close()
    return result

# 2. Vulnerable: Storing Plaintext Passwords
def store_password_vulnerable(username, password):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    connection.commit()
    connection.close()

# 3. Vulnerable: SQL Injection
def search_user_vulnerable(username):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM users WHERE username='{username}'")  # SQL Injection risk
    result = cursor.fetchall()
    connection.close()
    return result

# 4. Vulnerable: Exposing Debug Information
def debug_vulnerable():
    import os
    return os.popen('whoami').read()  # Revealing system information

# 5. Vulnerable: Hardcoded Credentials
def connect_to_db_vulnerable():
    return sqlite3.connect("database.db", uri=True)  # Hardcoded DB path

# 6. Vulnerable: Using Outdated Libraries
import flask  # Assume an old version with security flaws
def outdated_library_vulnerable():
    return "Running Flask version: " + flask.__version__

# 7. Vulnerable: Weak Session Token
def generate_session_token_vulnerable():
    import random
    return str(random.randint(1000, 9999))  # Easily guessable

# 8. Vulnerable: Loading Unverified Code
def load_external_script_vulnerable():
    exec(open("external_script.py").read())  # Unverified script execution

# 9. Vulnerable: Insufficient Logging
def login_vulnerable(username, password):
    print(f"User {username} attempted login")  # No security logs stored properly
    return "Login attempt recorded."

# 10. Vulnerable: Unvalidated URL Fetch
def fetch_url_vulnerable(url):
    response = requests.get(url)  # Open to SSRF attacks
    return response.text

# 11. Secure: Proper Access Control
def get_user_data_safe(user_id, current_user_id):
    if user_id != current_user_id:
        return "Access Denied"
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
    result = cursor.fetchone()
    connection.close()
    return result

# 12. Secure: Using Hashed Passwords (bcrypt)
def store_password_safe(username, password):
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    connection.commit()
    connection.close()

# 13. Secure: Using Parameterized Queries
def search_user_safe(username):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    result = cursor.fetchall()
    connection.close()
    return result

# 14. Secure: Removing Debugging
def debug_safe():
    return "Debugging is disabled in production."

# 15. Secure: Using Environment Variables
def connect_to_db_safe():
    db_path = os.getenv("DB_PATH", "database.db")
    return sqlite3.connect(db_path)

# 16. Secure: Regularly Updating Dependencies
def updated_library_safe():
    return "Using updated Flask version: " + flask.__version__

# 17. Secure: Using Secure Token Generation
def generate_session_token_safe():
    return secrets.token_hex(32)  # Cryptographically secure

# 18. Secure: Only Load Trusted Code
def load_external_script_safe():
    return "External scripts must be verified before execution."

# 19. Secure: Using Proper Logging
logging.basicConfig(filename='security.log', level=logging.INFO)
def login_safe(username, success):
    logging.info(f"User {username} login {'successful' if success else 'failed'}")
    return "Login attempt recorded."

# 20. Secure: Validate URL Before Fetching
def fetch_url_safe(url):
    allowed_domains = ["example.com", "api.trusted.com"]
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.netloc not in allowed_domains:
        return "Access Denied"
    response = requests.get(url)
    return response.text
