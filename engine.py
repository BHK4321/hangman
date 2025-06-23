from hangmanAI import HangmanSolver1, simulate_hangman_game

solver1 = HangmanSolver1(model_path= "best_model1.pth")
solver2 = HangmanSolver1(model_path= "best_model2.pth")

result = simulate_hangman_game(solver1, solver2, "bhaskar", verbose=True)