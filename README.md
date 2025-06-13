# 🔠 AI-Powered Hangman Game 🧠🎮

This project is an **interactive Hangman game** built with **Tkinter**, powered by two AI-based solvers that predict the next letter based on the current word state. The solvers adapt their strategy based on how much of the word is already revealed.

50% accuracy!!!
## 🧹 Features

* 🧠 **AI prediction** using two models:

  * `solver1` is used when more than 50% of the word is revealed.
  * `solver2` is used otherwise.
* 🔤 Real-time **letter prediction and display**.
* ❌ Limited number of incorrect guesses (with hangman animation).
* ✅ User inputs letter positions after every AI guess.

## 📁 File Structure

```
hangman_game.ipynb        # Jupyter Notebook containing the complete game and models
solver1 / solver2         # Pre-trained letter prediction models (must be loaded in notebook)
```

## 🚀 How to Run

1. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook hangman_game.ipynb
   ```

2. **Run all cells** in the notebook sequentially:

   * Model classes (`EnhancedHangmanModel`, `HangmanSolver`)
   * Training logic (optional)
   * UI logic with Tkinter

3. **Play the Game**:

   * The app will prompt you to enter the **target word** (hidden from the AI).
   * The model will make guesses.
   * You’ll manually input which positions the guessed letter appears in (1-based indexing).
   * The UI updates the hangman scaffold and word pattern.
   * The game ends when the word is guessed or attempts run out.

## 🛠️ Requirements

* Python 3.8+
* Tkinter (usually comes with Python)
* PyTorch
* Jupyter Notebook

Install requirements with:

```bash
pip install torch tqdm
```

## 🧠 Model Strategy

The models are trained on Hangman patterns and choose the next letter based on:

* Current revealed letters.
* Character adjacency (left/right).
* Word length, blank count, and vowel proximity.

📌 `solver1` → High-context guessing (≥50% known)
📌 `solver2` → Low-context guessing (<50% known)
