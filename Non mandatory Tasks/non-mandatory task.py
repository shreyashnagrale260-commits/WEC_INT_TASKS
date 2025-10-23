import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math

df = pd.read_csv("Untitled spreadsheet - Sheet1.csv")

for col in df.columns[2:]:
    df[col] = df[col].map({'yes': 1, 'no': 0})

characters = df['name'].tolist()
features = df.columns[2:]

def initialize_probabilities():
    return np.ones(len(characters)) / len(characters)

def update_probabilities(probs, q, ans):
    if ans not in ["yes", "no"]:
        return probs 
    ans_val = 1 if ans == "yes" else 0
    likelihoods = (df[q] == ans_val).astype(int)
    probs = probs * likelihoods
    if probs.sum() == 0:
        probs[:] = 1  
    return probs / probs.sum()

def question_information_gain(probs):
    info = {}
    for q in features:
        p_yes = np.dot(probs, df[q])  
        p_no = 1 - p_yes
        if p_yes in [0,1]: 
            continue
        entropy = - (p_yes * math.log2(p_yes) + p_no * math.log2(p_no))
        info[q] = entropy
    if not info:
        return None
    return sorted(info.items(), key=lambda x: x[1], reverse=True)[0][0]

# Get best guess so far
def best_guess(probs):
    idx = np.argmax(probs)
    return characters[idx], probs[idx]


def add_new_character():
    print("\n🔹 Let's add a new character to improve my knowledge!")
    name = input("Enter character name: ")
    movie = input("Enter movie name: ")

    answers = {}
    for q in features:
        ans = input(f"{q}? (yes/no): ").lower()
        if ans not in ["yes", "no"]:
            ans = "no"
        answers[q] = ans

    new_row = {"name": name, "movie": movie}
    new_row.update(answers)
    df.loc[len(df)] = new_row
    df.to_csv("indian_movie_characters.csv", index=False)
    print("New character added successfully!")
    

def play_game():
    print("Welcome to Akinator Clone: Indian Movie Characters Edition 🎬")
    print("Answer the questions with: yes / no / dont know")

    probs = initialize_probabilities()
    asked_questions = []

    for i in range(20):  
        q = question_information_gain(probs)
        if q is None or q in asked_questions:
            break

        asked_questions.append(q)
        print(f"\nQ{i+1}: Is your character {q.replace('_', ' ')}?")
        ans = input(" Your answer (yes/no/dont know): ").lower()

        probs = update_probabilities(probs, q, ans)
        name, confidence = best_guess(probs)

        print(f"\n My current best guess: {name} ({confidence*100:.2f}% confident)")

        if confidence > 0.8:
            print(f"\n I am {round(confidence*100)}% sure your character is {name}!")
            correct = input("Am I correct? (yes/no): ").lower()
            if correct == "yes":
                print(" Yay! I guessed it right!")
                return
            else:
                break

    print("\n I couldn’t guess your character.")
    choice = input("Would you like to teach me this new character? (yes/no): ").lower()
    if choice == "yes":
        add_new_character()


if __name__ == "__main__":
    play_game()
