import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
from pathlib import Path

from hangman_model import HangmanNN

MAX_WORD_LENGTH = 20
NUM_LETTERS = 26
NUM_ADDITIONAL_FEATURES = 2  # word length and num guesses
FEATURE_SIZE = MAX_WORD_LENGTH + NUM_LETTERS + NUM_ADDITIONAL_FEATURES


def prepare_features(word_pattern, guessed_letters, device):
    """Convert current game state to feature vector with consistent size"""
    features = []

    # Word pattern features (removing spaces)
    clean_pattern = word_pattern[::2] if isinstance(word_pattern, str) else word_pattern
    pattern_len = len(clean_pattern)

    # One-hot encode the pattern with padding
    pattern_features = []
    for pos in range(min(pattern_len, MAX_WORD_LENGTH)):
        if pos < len(clean_pattern):
            pattern_features.append(1 if clean_pattern[pos] != '_' else 0)
        else:
            pattern_features.append(0)

    # Pad pattern features if needed
    pattern_features.extend([0] * (MAX_WORD_LENGTH - len(pattern_features)))
    features.extend(pattern_features)

    # Guessed letters features (always 26 features)
    for letter in string.ascii_lowercase:
        features.append(1 if letter in guessed_letters else 0)

    # Add word length feature (normalized)
    features.append(pattern_len / MAX_WORD_LENGTH)

    # Add number of guesses made feature
    features.append(len(guessed_letters) / 6.0)

    assert len(features) == FEATURE_SIZE, f"Feature size mismatch: {len(features)} != {FEATURE_SIZE}"
    return torch.FloatTensor(features).to(device)


def generate_training_data(dictionary, device):
    """Generate training data from dictionary"""
    X, y = [], []

    print("Generating training examples...")
    total_words = len(dictionary)

    for i, word in enumerate(dictionary):
        if i % 10000 == 0:
            print(f"Processing word {i}/{total_words}...")

        if len(word) > MAX_WORD_LENGTH:
            continue

        # Generate multiple training examples per word
        for _ in range(3):
            guessed = []
            pattern = ['_'] * len(word)

            # Simulate random guesses
            num_guesses = np.random.randint(0, 5)
            for _ in range(num_guesses):
                letter = np.random.choice(list(string.ascii_lowercase))
                if letter not in guessed:
                    guessed.append(letter)
                    # Reveal letter in pattern if it exists
                    for i, char in enumerate(word):
                        if char == letter:
                            pattern[i] = letter

            features = prepare_features(pattern, guessed, device)

            # Create target vector (next best letter to guess)
            target = np.zeros(26)
            remaining_letters = set(word) - set(guessed)
            if remaining_letters:
                for letter in remaining_letters:
                    idx = ord(letter) - ord('a')
                    target[idx] = 1.0 / len(remaining_letters)

            X.append(features)
            y.append(torch.FloatTensor(target).to(device))

    return torch.stack(X), torch.stack(y)


def train_hangman_model(dictionary_path, model_save_path):
    """Train the model and save it to disk"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dictionary
    with open(dictionary_path, 'r') as f:
        dictionary = f.read().splitlines()

    # Generate training data
    X, y = generate_training_data(dictionary, device)

    # Initialize model with correct input size
    model = HangmanNN(input_size=FEATURE_SIZE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Training model...")
    batch_size = 128
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(X) // batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # Save the model
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'loss': avg_loss
    }, save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    train_hangman_model(
        dictionary_path="words_250000_train.txt",
        model_save_path="models/hangman_model.pt"
    )