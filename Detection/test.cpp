#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <mutex>
#include <unistd.h>

using namespace std;

// 1. Format String Vulnerability
void formatStringVuln() {
    char userInput[100];
    cout << "Enter a string: ";
    cin >> userInput;
    printf(userInput); // Vulnerable: Uncontrolled format string
    cout << endl;
}

// 2. Double Free Vulnerability
void doubleFreeVuln() {
    int* ptr = (int*)malloc(sizeof(int));
    if (!ptr) return;
    *ptr = 10;
    free(ptr);
    free(ptr); // Vulnerable: Double free
}

// 3. Race Condition in File Access
void raceConditionVuln() {
    ofstream file("temp.txt");
    file << "Sensitive Data";
    file.close();
    remove("temp.txt"); // Vulnerable: Race condition
}

// 4. Buffer Overflow Vulnerability
void bufferOverflowVuln() {
    char buffer[10];
    strcpy(buffer, "ThisIsAVeryLongString"); // Vulnerable: Buffer overflow
    cout << buffer << endl;
}

// 5. Use After Free Vulnerability
void useAfterFreeVuln() {
    int* ptr = (int*)malloc(sizeof(int));
    *ptr = 42;
    free(ptr);
    cout << *ptr << endl; // Vulnerable: Use after free
}

// 6. Command Injection Vulnerability
void commandInjectionVuln() {
    char command[100];
    cout << "Enter command: ";
    cin >> command;
    system(command); // Vulnerable: Arbitrary command execution
}

// 7. Insecure Deserialization
void insecureDeserializationVuln() {
    char input[100];
    cout << "Enter serialized data: ";
    cin >> input;
    char* data = new char[strlen(input)];
    strcpy(data, input); // Vulnerable: No validation
    delete[] data;
}

// 8. Hardcoded Credentials
void hardcodedCredentialsVuln() {
    string username = "admin";
    string password = "password123"; // Vulnerable: Hardcoded credentials
    cout << "Username: " << username << " Password: " << password << endl;
}

// 9. Integer Overflow
void integerOverflowVuln() {
    unsigned int num = UINT_MAX;
    num += 1; // Vulnerable: Integer overflow
    cout << "Number: " << num << endl;
}

// 10. Improper Input Validation
void improperInputValidationVuln() {
    int age;
    cout << "Enter your age: ";
    cin >> age;
    if (age < 0) {
        cout << "Invalid age!" << endl;
    }
}

// 11. Safe Format String Usage
void safeFormatString() {
    char userInput[100];
    cout << "Enter a string: ";
    cin >> userInput;
    printf("%s", userInput); // Safe: Controlled format string
    cout << endl;
}

// 12. Safe Memory Management
void safeMemoryManagement() {
    int* ptr = (int*)malloc(sizeof(int));
    if (!ptr) return;
    *ptr = 10;
    free(ptr);
    ptr = nullptr; // Safe: Nullify after free
}

// 13. Thread-safe File Operation
void safeFileOperation() {
    const char* filename = "tempfile.txt";
    {
        ofstream file(filename);
        if (file.is_open()) {
            file << "Safe content!";
            file.close();
            cout << "File written safely!" << endl;
        }
    }
}

// 14. Thread-safe Counter
void threadSafeCounter() {
    static mutex mtx;
    static int counter = 0;
    {
        lock_guard<mutex> lock(mtx);
        counter++;
        cout << "Counter: " << counter << endl;
    }
}

// 15. Safe Command Execution
void safeCommandExecution() {
    cout << "Executing predefined command safely." << endl;
    system("ls"); // Safe: Controlled execution
}

// 16. Secure Password Handling
void securePasswordHandling() {
    string password;
    cout << "Enter password: ";
    cin >> password;
    cout << "Password securely received." << endl;
}

// 17. Safe Integer Handling
void safeIntegerHandling() {
    unsigned int num = UINT_MAX - 1;
    num += 1; // Safe: Checked boundary
    cout << "Number: " << num << endl;
}

// 18. Safe Input Validation
void safeInputValidation() {
    int age;
    cout << "Enter your age: ";
    cin >> age;
    if (age < 0 || age > 120) {
        cout << "Invalid age!" << endl;
    } else {
        cout << "Valid age entered." << endl;
    }
}

// 19. Secure File Access
void secureFileAccess() {
    ifstream file("secure_file.txt");
    if (file.is_open()) {
        cout << "Secure file access granted." << endl;
        file.close();
    } else {
        cout << "File not found." << endl;
    }
}

// 20. Safe Deserialization
void safeDeserialization() {
    char input[100];
    cout << "Enter data: ";
    cin >> input;
    if (strlen(input) < 50) { // Safe: Checking input size
        char* data = new char[strlen(input)];
        strcpy(data, input);
        cout << "Data safely deserialized." << endl;
        delete[] data;
    }
}
