<?php
// Example 1: PHP without input sanitisation, vulnerable to XSS
// Flagged by transformer model as vulnerable
function processAndDisplay($data) {
    if (!isset($data['name']) || !isset($data['comment'])) {
        echo "Missing required information.";
        return;
    }

    $name = $data['name'];
    $comment = $data['comment'];

    $htmlOutput  = "<div class='comment-section'>";
    $htmlOutput .= "<h3>User: $name</h3>";
    $htmlOutput .= "<p>Comment: $comment</p>";
    $htmlOutput .= "</div>";

    echo $htmlOutput;
}
?>

<?php
// Example 2: PHP with input sanitisation afterwards, not vulnerable to XSS
// Flagged by transformer model as vulnerable
function processAndDisplay($data) {
    if (!isset($data['name']) || !isset($data['comment'])) {
        echo "Missing required information.";
        return;
    }

    $name = trim($data['name']);
    $comment = trim($data['comment']);

    $name = htmlspecialchars($name, ENT_QUOTES, 'UTF-8');
    $comment = htmlspecialchars($comment, ENT_QUOTES, 'UTF-8');

    $htmlOutput  = "<div class='comment-section'>";
    $htmlOutput .= "<h3>User: $name</h3>";
    $htmlOutput .= "<p>Comment: $comment</p>";
    $htmlOutput .= "</div>";

    echo $htmlOutput;
}
?>

<?php
// Example 3: PHP with proper input sanitisation, not vulnerable to XSS
// Flagged by transformer model as not vulnerable
function processAndDisplay($data) {
    if(isset($_GET['name'])) {
        $name = filter_input(INPUT_GET, 'name', FILTER_SANITIZE_STRING);
        echo "Hello, ".htmlspecialchars($name, ENT_QUOTES, 'UTF-8');
    } else {
        echo "Please provide your name.";
    }
}
?>