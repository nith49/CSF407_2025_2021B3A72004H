import numpy as np
import json
import matplotlib.pyplot as plt
import requests

# Replace with actual API keys before running
GROQ_API_KEY = ""
GEMINI_API_KEY = ""

def groq_api_call(prompt, api_key):
    url = "https://api.groq.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 10
    })
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return "Error"

def gemini_api_call(prompt, api_key):
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"prompt": {"text": prompt}, "temperature": 0.7})
    response = requests.post(f"{url}?key={api_key}", headers=headers, data=payload)
    if response.status_code == 200:
        return response.json().get("candidates", [{}])[0].get("output", "Error").strip()
    else:
        return "Error"

def initialize_board(n):
    return np.full((n, n), '', dtype=str)

def print_board(board):
    for row in board:
        print(" | ".join([cell if cell else " " for cell in row]))
        print("-" * (4 * len(board) - 1))

def generate_llm_prompt(board, last_move, llm_agent):
    prompt = f"You are {llm_agent}. The current board state is:\n"
    for row in board:
        prompt += " | ".join(cell if cell else "-" for cell in row) + "\n"
    if last_move:
        prompt += f"The opponent's last move was at position {last_move}.\n"
    prompt += "Provide the best next move as row,col."
    return prompt

def get_llm_move(board, last_move, player):
    prompt = generate_llm_prompt(board, last_move, player)
    if player == "LLM1":
        move_str = groq_api_call(prompt, GROQ_API_KEY)
    else:
        move_str = gemini_api_call(prompt, GEMINI_API_KEY)
    try:
        move = tuple(map(int, move_str.split(",")))
        if board[move] == '':
            return move
    except:
        pass
    return np.random.choice(len(board)), np.random.choice(len(board))

def get_next_move(board, last_move, player, is_human=False):
    if is_human:
        while True:
            try:
                move = tuple(map(int, input("Enter your move (row,col): ").split(',')))
                if board[move] == '':
                    return move
                else:
                    print("Cell already occupied. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Enter row,col within board range.")
    else:
        return get_llm_move(board, last_move, player)

def check_winner(board, symbol):
    n = len(board)
    for row in board:
        if all(cell == symbol for cell in row):
            return True
    for col in range(n):
        if all(board[row][col] == symbol for row in range(n)):
            return True
    if all(board[i][i] == symbol for i in range(n)) or all(board[i][n-i-1] == symbol for i in range(n)):
        return True
    return False

def play_game(n, player1, player2, human_player=None):
    board = initialize_board(n)
    players = [player1, player2]
    symbols = ["X", "O"]
    last_move = None
    
    for turn in range(n * n):
        player = players[turn % 2]
        symbol = symbols[turn % 2]
        is_human = player == human_player
        
        move = get_next_move(board, last_move, player, is_human)
        while board[move] != '':
            move = get_next_move(board, last_move, player, is_human)
        
        board[move] = symbol
        last_move = move
        print(f"{player} plays at {move}")
        print_board(board)
        
        if check_winner(board, symbol):
            return player
    
    return player2  # If draw, LLM1 loses

def run_trials(n, trials=50):
    results = {"LLM1": 0, "LLM2": 0}
    for _ in range(trials):
        winner = play_game(n, "LLM1", "LLM2")
        results[winner] += 1
    
    with open("Exercise1.json", "w") as file:
        json.dump(results, file)
    
    return results

def plot_binomial_distribution(results):
    labels = list(results.keys())
    values = list(results.values())
    
    plt.bar(labels, values, color=["blue", "orange"])
    plt.xlabel("Game Outcome")
    plt.ylabel("Frequency")
    plt.title("Binomial Distribution of Tic-Tac-Toe Wins (50 Trials)")
    plt.savefig("Exercise1.png")
    plt.show()

if __name__ == "_main_":
    mode = input("Choose mode: (1) LLM vs LLM (2) Human vs LLM: ")
    board_size = int(input("Enter board size (NxN): "))
    if mode == "1":
        results = run_trials(board_size, 50)
        plot_binomial_distribution(results)
    else:
        human_symbol = input("Choose your symbol (X or O): ").upper()
        human_player = "Human"
        llm_player = "LLM1" if human_symbol == "O" else "LLM2"
        winner = play_game(board_size, human_player, llm_player, human_player)
        print(f"Winner: {winner}")