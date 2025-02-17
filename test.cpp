#include <iostream>
#include <string>

// Function for SQL Injection vulnerability (simulated)
void vulnerableSQLLogin(const std::string& username, const std::string& password) {
    // Vulnerable SQL query (SQL Injection)
    std::string query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";

    std::cout << "Executing SQL Query: " << query << std::endl;
    
    // Simulate a successful login (no actual database interaction)
    if (username == "admin" && password == "admin123") {
        std::cout << "Login successful (Vulnerable SQL)" << std::endl;
    } else {
        std::cout << "Invalid username or password" << std::endl;
    }
}

// Function for XSS vulnerability
void vulnerableXSS(const std::string& userInput) {
    std::cout << "Your comment: " << userInput << std::endl;  // This is vulnerable to XSS if input contains HTML tags or JS code
}


// Function for SQL Injection protection (simulated)
void nonVulnerableSQLLogin(const std::string& username, const std::string& password) {
    // Non-vulnerable query using prepared statements (simulated)
    std::cout << "Executing SQL Query (protected): SELECT * FROM users WHERE username = ? AND password = ?" << std::endl;

    // Simulate a successful login (no actual database interaction)
    if (username == "admin" && password == "admin123") {
        std::cout << "Login successful (Non-vulnerable SQL)" << std::endl;
    } else {
        std::cout << "Invalid username or password" << std::endl;
    }
}


// Function for XSS protection
void nonVulnerableXSS(const std::string& userInput) {
    std::string safeInput = userInput;
    for (size_t i = 0; i < safeInput.length(); i++) {
        if (safeInput[i] == '<') {
            safeInput.replace(i, 1, "&lt;");
        } else if (safeInput[i] == '>') {
            safeInput.replace(i, 1, "&gt;");
        } else if (safeInput[i] == '&') {
            safeInput.replace(i, 1, "&amp;");
        } else if (safeInput[i] == '"') {
            safeInput.replace(i, 1, "&quot;");
        }
    }
    std::cout << "Your comment (safe): " << safeInput << std::endl;  // Output encoded to prevent XSS
}

int main() {
    std::string username, password, comment;

    // Example for SQL Injection vulnerability
    std::cout << "\nVulnerable SQL Login (SQL Injection):\n";
    std::cout << "Enter username: ";
    std::cin >> username;
    std::cout << "Enter password: ";
    std::cin >> password;
    vulnerableSQLLogin(username, password);

    // Example for non-vulnerable SQL Login
    std::cout << "\nNon-Vulnerable SQL Login (Prepared Statements):\n";
    std::cout << "Enter username: ";
    std::cin >> username;
    std::cout << "Enter password: ";
    std::cin >> password;
    nonVulnerableSQLLogin(username, password);

    // Example for XSS vulnerability
    std::cout << "\nVulnerable XSS:\n";
    std::cout << "Enter your comment: ";
    std::cin.ignore();  // to clear the buffer from previous inputs
    std::getline(std::cin, comment);
    vulnerableXSS(comment);

    // Example for non-vulnerable XSS
    std::cout << "\nNon-Vulnerable XSS:\n";
    std::cout << "Enter your comment: ";
    std::getline(std::cin, comment);
    nonVulnerableXSS(comment);

    return 0;
}
