<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PHP Vulnerability Example</title>
</head>
<body>

    <h1>PHP Vulnerability Example</h1>

    <h2>Vulnerable Code 1: SQL Injection</h2>
    <form method="POST">
        <label for="username">Username:</label>
        <input type="text" name="username" id="username" required>
        <label for="password">Password:</label>
        <input type="password" name="password" id="password" required>
        <button type="submit">Login</button>
    </form>

    <?php
    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
        $username = $_POST['username'];
        $password = $_POST['password'];

        // Vulnerable SQL query (SQL Injection)
        $conn = new mysqli("localhost", "root", "", "testdb");

        if ($conn->connect_error) {
            die("Connection failed: " . $conn->connect_error);
        }

        $sql = "SELECT * FROM users WHERE username = '$username' AND password = '$password'";
        $result = $conn->query($sql);

        if ($result->num_rows > 0) {
            echo "<p>Login successful</p>";
        } else {
            echo "<p>Invalid username or password</p>";
        }

        $conn->close();
    }
    ?>

<h2>Vulnerable Code 2: Cross-Site Scripting (XSS)</h2>
    <form method="POST">
        <label for="comment">Leave a comment:</label>
        <textarea name="comment" id="comment" required></textarea>
        <button type="submit">Submit</button>
    </form>

    <?php
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['comment'])) {
        $comment = $_POST['comment'];

        // Vulnerable XSS: Directly echoing user input
        echo "<h3>Your Comment:</h3>";
        echo "<p>$comment</p>";  // This is vulnerable to XSS if a user enters a script tag
    }
    ?>

    

    <h2>Non-Vulnerable Code 1: Prepared Statements</h2>
    <form method="POST">
        <label for="username2">Username:</label>
        <input type="text" name="username2" id="username2" required>
        <label for="password2">Password:</label>
        <input type="password" name="password2" id="password2" required>
        <button type="submit">Login</button>
    </form>

    <?php
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['username2']) && isset($_POST['password2'])) {
        $username2 = $_POST['username2'];
        $password2 = $_POST['password2'];

        // Non-vulnerable SQL query using prepared statements
        $conn = new mysqli("localhost", "root", "", "testdb");

        if ($conn->connect_error) {
            die("Connection failed: " . $conn->connect_error);
        }

        $stmt = $conn->prepare("SELECT * FROM users WHERE username = ? AND password = ?");
        $stmt->bind_param("ss", $username2, $password2); // "ss" denotes the type of the parameters (string, string)
        $stmt->execute();
        $result = $stmt->get_result();

        if ($result->num_rows > 0) {
            echo "<p>Login successful</p>";
        } else {
            echo "<p>Invalid username or password</p>";
        }

        $stmt->close();
        $conn->close();
    }
    ?>

    

    <h2>Non-Vulnerable Code 2: Output Encoding</h2>
    <form method="POST">
        <label for="comment2">Leave a comment:</label>
        <textarea name="comment2" id="comment2" required></textarea>
        <button type="submit">Submit</button>
    </form>

    <?php
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['comment2'])) {
        $comment2 = $_POST['comment2'];

        // Non-vulnerable XSS: Proper output encoding
        echo "<h3>Your Comment:</h3>";
        echo "<p>" . htmlspecialchars($comment2, ENT_QUOTES, 'UTF-8') . "</p>";  // Safe output encoding
    }
    ?>

</body>
</html>
