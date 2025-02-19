#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <unistd.h>

using namespace std;

// 1. Format String Vulnerability
void formatStringVuln() {
    char userInput[100];
    cout << "Enter a string: ";
    cin >> userInput;
    printf(userInput);  // Dangerous: Unchecked format string
    cout << endl;
}

// 2. Double Free Vulnerability
void doubleFreeVuln() {
    int* ptr = (int*)malloc(sizeof(int));
    if (!ptr) return;
    *ptr = 10;
    free(ptr);
    free(ptr);  // Double free, could lead to memory corruption
}

// 3. File Operation without Race Condition (Non-Vulnerable)
void safeFileOperation() {
    const char* filename = "tempfile.txt";
    
    // Open the file with exclusive lock to avoid race condition
    ofstream file(filename);
    if (file.is_open()) {
        file << "Safe content!";
        file.close();
        cout << "File written safely!" << endl;
    }
    else {
        cout << "Failed to open the file." << endl;
    }
}
// 3. File Operation without Race Condition (Non-Vulnerable)
void safeFileOperation() {
    const char* filename = "tempfile.txt";
    
    // Open the file with exclusive lock to avoid race condition
    ofstream file(filename);
    if (file.is_open()) {
        file << "Safe content!";
        file.close();
        cout << "File written safely!" << endl;
    }
    else {
        cout << "Failed to open the file." << endl;
    }
}

int main() {
    formatStringVuln();
    doubleFreeVuln();
    safeFileOperation();  // This one is non-vulnerable
    return 0;
}
