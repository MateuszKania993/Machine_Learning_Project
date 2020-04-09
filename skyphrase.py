"""Script for checking number of correct password in the file."""


# Function that checks if the word in the string does not repeat.
# If evry word is unique returns True, else return False
def check_password(password: str) -> bool:
    list_of_words = []
    password_list = password.rstrip('\n').split(' ')  # Turn string to a list of words and remove '\n' from end.
    for word in password_list:
        if word in list_of_words:  # Check if the word already in the list.
            return False
        else:
            list_of_words.append(word)
    return True


number_correct_passwords = 0

# Opening a file with the password, and passing ery line to check_password function.
with open("skychallenge_skyphrase_input.txt", "r") as file:
    for line in file:
        if check_password(line):
            number_correct_passwords += 1

# Printing the final result
print(f"Number of correct skyphrases is: {number_correct_passwords}")
