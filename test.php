<?php
// 1. Vulnerable: Insecure Direct Object Reference (IDOR)
function getUserDataVulnerable($userId, $conn) 
{
    $query = "SELECT * FROM users WHERE id=$userId"; // No authorization check!
    $result = mysqli_query($conn, $query);
    
    return mysqli_fetch_assoc($result);
}
?>
<?php
// 2. Vulnerable: SQL Injection
function searchUserVulnerable($username, $conn) 
{
    $query = "SELECT * FROM users WHERE username = '$username'"; // Vulnerable to SQL Injection!
    $result = mysqli_query($conn, $query);
    
    return mysqli_fetch_assoc($result);
}
?>
<?php
// 3. Vulnerable: Cross-Site Scripting (XSS)
function displayCommentVulnerable($comment) 
{
    echo "<p>Comment: $comment</p>"; // Outputting user input directly
}
?>
<?php
// 4. Vulnerable: Cross-Site Request Forgery (CSRF)
function updateEmailVulnerable($userId, $newEmail, $conn) 
{
    $query = "UPDATE users SET email='$newEmail' WHERE id=$userId"; // No CSRF protection!
    mysqli_query($conn, $query);
}
?>
<?php
// 5. Vulnerable: Insecure Password Storage
function storePasswordVulnerable($password, $conn) 
{
    $query = "INSERT INTO users (password) VALUES ('$password')"; // Storing passwords in plaintext!
    mysqli_query($conn, $query);
}
?>
<?php
// 6. Vulnerable: Unvalidated File Upload
function uploadFileVulnerable($file) 
{
    move_uploaded_file($file['tmp_name'], "uploads/" . $file['name']); // No file type validation!
}
?>
<?php
// 7. Vulnerable: Security Misconfiguration (Exposing PHP Info)
function showPhpInfoVulnerable() 
{
    phpinfo(); // Exposes sensitive system details
}
?>
<?php
// 8. Vulnerable: Insecure Deserialization
function deserializeDataVulnerable($data) 
{
    return unserialize($data); // Allows malicious object injection!
}
?>
<?php
// 9. Vulnerable: Using Hardcoded API Keys
function connectToApiVulnerable() 
{
    $apiKey = "1234567890abcdef"; // Hardcoded sensitive key!
    return "https://api.example.com/data?key=" . $apiKey;
}
?>
<?php
// 10. Vulnerable: Server-Side Request Forgery (SSRF)
function fetchUrlVulnerable($url) 
{
    return file_get_contents($url); // No validation of external URLs!
}
?>
<?php
// 11. Secure: Proper Access Control for IDOR
function getUserDataSafe($userId, $conn, $currentUser) 
{
    if ($userId !== $currentUser) {
        die("Unauthorized access!"); // Prevents unauthorized access
    }

    $stmt = $conn->prepare("SELECT * FROM users WHERE id = ?");
    $stmt->bind_param("i", $userId);
    $stmt->execute();
    
    return $stmt->get_result()->fetch_assoc();
}
?>
<?php
// 12. Secure: Using Parameterized Queries (Prevent SQL Injection)
function searchUserSafe($username, $conn) 
{
    $stmt = $conn->prepare("SELECT * FROM users WHERE username = ?");
    $stmt->bind_param("s", $username);
    $stmt->execute();
    
    return $stmt->get_result()->fetch_assoc();
}
?>
<?php
// 13. Secure: Escaping Output to Prevent XSS
function displayCommentSafe($comment) 
{
    echo "<p>Comment: " . htmlspecialchars($comment, ENT_QUOTES, 'UTF-8') . "</p>";
}
?>
<?php
// 14. Secure: CSRF Protection with Token
function updateEmailSafe($userId, $newEmail, $conn, $csrfToken, $sessionToken) 
{
    if ($csrfToken !== $sessionToken) {
        die("CSRF detected!");
    }

    $stmt = $conn->prepare("UPDATE users SET email = ? WHERE id = ?");
    $stmt->bind_param("si", $newEmail, $userId);
    $stmt->execute();
}
?>
<?php
// 15. Secure: Hashing Passwords with Bcrypt
function storePasswordSafe($password, $conn) 
{
    $hashedPassword = password_hash($password, PASSWORD_BCRYPT);

    $stmt = $conn->prepare("INSERT INTO users (password) VALUES (?)");
    $stmt->bind_param("s", $hashedPassword);
    $stmt->execute();
}
?>
<?php
// 16. Secure: Validating File Uploads
function uploadFileSafe($file) 
{
    $allowedTypes = ["image/png", "image/jpeg"];
    
    if (!in_array($file['type'], $allowedTypes)) {
        die("Invalid file type!");
    }

    move_uploaded_file($file['tmp_name'], "uploads/" . basename($file['name']));
}
?>
<?php
// 17. Secure: Restricting PHP Info Access
function showPhpInfoSafe() 
{
    if ($_SERVER['REMOTE_ADDR'] === '127.0.0.1') {
        phpinfo(); // Show info only to local admin
    } else {
        die("Access denied.");
    }
}
?>
<?php
// 18. Secure: Using JSON for Safe Serialization
function deserializeDataSafe($data) 
{
    return json_decode($data, true); // Prevents object injection attacks
}
?>
<?php
// 19. Secure: Using Environment Variables for API Keys
function connectToApiSafe() 
{
    $apiKey = getenv('API_KEY'); // Fetch from environment variables
    return "https://api.example.com/data?key=" . $apiKey;
}
?>
<?php
// 20. Secure: Validating External URLs to Prevent SSRF
function fetchUrlSafe($url) 
{
    $allowedDomains = ["example.com", "trusted.com"];

    $parsedUrl = parse_url($url);
    if (!in_array($parsedUrl['host'], $allowedDomains)) {
        die("Unauthorized URL request!");
    }

    return file_get_contents($url);
}
?>
