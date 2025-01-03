import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
import random
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm

from hangman_model import HangmanNN

# Constants
MAX_WORD_LENGTH = 20
NUM_LETTERS = len(string.ascii_lowercase)  # 26

# Calculate feature size components
PATTERN_FEATURES = MAX_WORD_LENGTH        # One-hot word pattern
GUESSED_FEATURES = NUM_LETTERS            # Letter presence
NGRAM_FEATURES = NUM_LETTERS              # N-gram based scores
LETTER_FREQ_FEATURES = NUM_LETTERS        # Letter frequencies
STRUCTURE_FEATURES = 2                    # Word length and num guesses
RATIO_FEATURES = 2                        # Vowel and consonant ratios

# Total feature size
FEATURE_SIZE = (
        PATTERN_FEATURES +    # 20
        GUESSED_FEATURES +    # 26
        NGRAM_FEATURES +      # 26
        LETTER_FREQ_FEATURES + # 26
        STRUCTURE_FEATURES +  # 2
        RATIO_FEATURES       # 2
)


class BalancedFeatureExtractor:
    def __init__(self, dictionary):
        print("Initializing Balanced Feature Extractor...")

        # Store feature sizes for verification
        self.feature_sizes = {
            'pattern': PATTERN_FEATURES,
            'guessed': GUESSED_FEATURES,
            'ngram': NGRAM_FEATURES,
            'letter_freq': LETTER_FREQ_FEATURES,
            'structure': STRUCTURE_FEATURES,
            'ratio': RATIO_FEATURES
        }

        # Pre-filter valid words
        self.dictionary = [word for word in dictionary if len(word) <= MAX_WORD_LENGTH]
        print(f"Dictionary size after filtering: {len(self.dictionary)} words")

        # Pre-compute letter positions for all words
        print("Pre-computing word letter positions...")
        self.word_letters = {}
        for word in tqdm(self.dictionary):
            self.word_letters[word] = {char: [i for i, c in enumerate(word) if c == char]
                                       for char in set(word)}

        # Build positional n-grams
        print("Building positional n-grams...")
        self.positional_ngrams = self._build_positional_ngrams()

        # Pre-compute word lengths
        self.word_lengths = {word: len(word) for word in self.dictionary}

        # Build letter frequency table
        all_letters = "".join(self.dictionary)
        total_letters = len(all_letters)
        self.letter_frequencies: dict[str, float] = {}
        for letter in string.ascii_lowercase:
            self.letter_frequencies[letter] = all_letters.count(letter) / total_letters

    def _build_positional_ngrams(self):
        """Build position-aware n-gram frequencies"""
        from collections import defaultdict

        # Structure: {position: {ngram: {next_letter: count}}}
        ngram_data = defaultdict(lambda: defaultdict(Counter))

        for word in tqdm(self.dictionary, desc="Computing n-grams"):
            # For each position in the word
            for pos in range(len(word) - 1):
                # For each n-gram size (1 and 2 for efficiency)
                for n in range(1, min(3, len(word) - pos)):
                    ngram = word[pos:pos + n]
                    next_letter = word[pos + n]
                    ngram_data[pos][ngram][next_letter] += 1

        # Normalize counts to probabilities
        normalized_data = {}
        for pos in ngram_data:
            normalized_data[pos] = {}
            for ngram, next_counts in ngram_data[pos].items():
                total = sum(next_counts.values())
                normalized_data[pos][ngram] = {
                    letter: count/total
                    for letter, count in next_counts.items()
                }

        return normalized_data

    def get_ngram_scores(self, pattern, guessed_letters):
        """Get letter scores based on positional n-grams"""
        scores = Counter()

        # Get position of each underscore
        unknown_positions = [i for i, char in enumerate(pattern) if char == '_']

        for pos in unknown_positions:
            # Look at previous letter(s) if available
            if pos > 0:
                prev_letter = pattern[pos-1]
                if prev_letter != '_':
                    # Single letter context
                    if pos-1 in self.positional_ngrams and prev_letter in self.positional_ngrams[pos-1]:
                        next_probs = self.positional_ngrams[pos-1][prev_letter]
                        for letter, prob in next_probs.items():
                            if letter not in guessed_letters:
                                scores[letter] += prob

                    # Two letter context if available
                    if pos > 1 and pattern[pos-2] != '_':
                        bigram = pattern[pos-2:pos]
                        if pos-2 in self.positional_ngrams and bigram in self.positional_ngrams[pos-2]:
                            next_probs = self.positional_ngrams[pos-2][bigram]
                            for letter, prob in next_probs.items():
                                if letter not in guessed_letters:
                                    scores[letter] += prob * 2  # Weight bigrams more

        return scores

    def extract_features_balanced(self, word_pattern, guessed_letters, device):
        """Extract features with size verification"""
        features = []

        # Clean pattern
        if isinstance(word_pattern, list):
            clean_pattern = "".join(word_pattern)
        else:
            clean_pattern = word_pattern[::2]

        # Track feature counts for verification
        feature_counts = {key: 0 for key in self.feature_sizes.keys()}

        # 1. Pattern features (MAX_WORD_LENGTH)
        pattern_features = []
        for pos in range(MAX_WORD_LENGTH):
            if pos < len(clean_pattern):
                pattern_features.append(1 if clean_pattern[pos] != '_' else 0)
            else:
                pattern_features.append(0)
        features.extend(pattern_features)
        feature_counts['pattern'] = len(pattern_features)

        # 2. Guessed letters (NUM_LETTERS)
        guessed_features = [1 if letter in guessed_letters else 0
                            for letter in string.ascii_lowercase]
        features.extend(guessed_features)
        feature_counts['guessed'] = len(guessed_features)

        # 3. N-gram based scores (NUM_LETTERS)
        ngram_scores = self.get_ngram_scores(clean_pattern, guessed_letters)
        ngram_features = [ngram_scores[letter] for letter in string.ascii_lowercase]
        features.extend(ngram_features)
        feature_counts['ngram'] = len(ngram_features)

        # 4. Letter frequencies (NUM_LETTERS)
        matching_words = self.get_matching_words_fast(clean_pattern)
        if matching_words:
            all_letters = "".join(matching_words)
            total = len(all_letters)
            freq_features = [all_letters.count(letter) / total if total > 0 else 0
                             for letter in string.ascii_lowercase]
        else:
            freq_features = [self.letter_frequencies[letter]
                             for letter in string.ascii_lowercase]
        features.extend(freq_features)
        feature_counts['letter_freq'] = len(freq_features)

        # 5. Word structure features (2)
        structure_features = [
            len(clean_pattern) / MAX_WORD_LENGTH,  # length
            len(guessed_letters) / 6.0             # num guesses
        ]
        features.extend(structure_features)
        feature_counts['structure'] = len(structure_features)

        # 6. Vowel/consonant ratios (2)
        remaining = clean_pattern.count('_')
        if remaining > 0:
            vowels = set('aeiou')
            consonants = set(string.ascii_lowercase) - vowels
            vowel_score = sum(ngram_scores[l] for l in vowels if l not in guessed_letters)
            consonant_score = sum(ngram_scores[l] for l in consonants if l not in guessed_letters)
            total_score = vowel_score + consonant_score
            if total_score > 0:
                vowel_ratio = vowel_score / total_score
            else:
                vowel_ratio = 0.4
        else:
            vowel_ratio = 0

        ratio_features = [vowel_ratio, 1 - vowel_ratio]
        features.extend(ratio_features)
        feature_counts['ratio'] = len(ratio_features)

        # Verify feature sizes
        for key, count in feature_counts.items():
            expected = self.feature_sizes[key]
            assert count == expected, f"Feature size mismatch for {key}: got {count}, expected {expected}"

        assert len(features) == FEATURE_SIZE, f"Total feature size mismatch: got {len(features)}, expected {FEATURE_SIZE}"

        return torch.FloatTensor(features).to(device)

    def get_matching_words_fast(self, pattern):
        """Faster word matching without regex"""
        pattern_len = len(pattern)
        matching = []

        for word in self.dictionary:
            if self.word_lengths[word] != pattern_len:
                continue

            matches = True
            for i, char in enumerate(pattern):
                if char != '_' and char != word[i]:
                    matches = False
                    break
            if matches:
                matching.append(word)

        return matching


def generate_training_data_fast(feature_extractor, device,
                                data_cache_path="data/training_data.pt",
                                max_examples=100000,
                                examples_per_word=5):
    """Generate training data with caching"""

    # Check for cached data
    cache_path = Path(data_cache_path)
    if cache_path.exists():
        print("Loading cached training data...")
        try:
            data = torch.load(cache_path, map_location=device)
            print(f"Loaded {len(data['X'])} cached examples")
            return data['X'], data['y'], data['feature_extractor']
        except Exception as e:
            print(f"Error loading cached data: {e}")
            print("Regenerating training data...")

    X, y = [], []

    # Pre-generate random states for all examples
    print("Generating random states...")
    all_words = [(word, i) for word in feature_extractor.dictionary
                 for i in range(examples_per_word)]
    random.shuffle(all_words)

    try:
        # Process in batches
        batch_size = 1000
        for i in tqdm(range(0, len(all_words), batch_size), desc="Generating training examples"):
            batch_words = all_words[i:i+batch_size]

            for word, _ in batch_words:
                guessed = []
                pattern = ['_'] * len(word)

                # Simulate game state
                num_guesses = np.random.randint(0, 5)
                letters_to_guess = random.sample(string.ascii_lowercase, num_guesses)

                for letter in letters_to_guess:
                    if letter not in guessed:
                        guessed.append(letter)
                        if letter in word:
                            for pos in feature_extractor.word_letters[word].get(letter, []):
                                pattern[pos] = letter

                # Extract features
                features = feature_extractor.extract_features_balanced(pattern, guessed, device)

                # Create target
                target = np.zeros(26)
                remaining_letters = set(word) - set(guessed)
                if remaining_letters:
                    for letter in remaining_letters:
                        weight = 1.0
                        if letter == word[0]: weight *= 1.5
                        if letter == word[-1]: weight *= 1.5
                        if letter in 'aeiou': weight *= 1.2
                        target[ord(letter) - ord('a')] = weight
                    target = target / target.sum()

                X.append(features)
                y.append(torch.FloatTensor(target).to(device))

                if len(X) >= max_examples:
                    print(f"\nReached maximum examples limit ({max_examples})")
                    raise StopIteration  # Use exception to break out of nested loops

    except StopIteration:
        pass  # Handle the max examples case gracefully
    finally:
        # Always stack and save the data, regardless of how we exited the loop
        if X and y:  # Make sure we have some data
            print(f"\nStacking {len(X)} examples...")
            X = torch.stack(X)
            y = torch.stack(y)

            print("Saving training data to cache...")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'X': X,
                'y': y,
                'feature_extractor': feature_extractor
            }, cache_path)
            print(f"Data cached to {cache_path}")
        else:
            print("No examples were generated!")

    return X, y, feature_extractor


def train_model(X, y, model_save_path, device):
    """Train the model with validation"""
    # Split into train/validation
    indices = torch.randperm(len(X))
    split = int(0.9 * len(X))
    train_indices = indices[:split]
    val_indices = indices[split:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # Initialize model and training
    model = HangmanNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    batch_size = 128
    epochs = 20
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    print("\nTraining model...")
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = tqdm(range(0, len(X_train), batch_size),
                             desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for i in train_batches:
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_batches)

        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = tqdm(range(0, len(X_val), batch_size),
                           desc=f"Epoch {epoch + 1}/{epochs} [Val]")

        with torch.no_grad():
            for i in val_batches:
                batch_X = X_val[i:i + batch_size]
                batch_y = y_val[i:i + batch_size]

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                val_batches.set_postfix({'loss': loss.item()})

        avg_val_loss = total_val_loss / len(val_batches)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss,
                'feature_size': FEATURE_SIZE
            }, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

    print("\nTraining completed!")
    return model


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    dictionary_path = "words_250000_train.txt"
    model_save_path = "models/hangman_model.pt"
    data_cache_path = "data/training_data.pt"

    # Load dictionary
    print("Loading dictionary...")
    with open(dictionary_path, 'r') as f:
        dictionary = f.read().splitlines()

    # Create feature extractor
    feature_extractor = BalancedFeatureExtractor(dictionary)

    # Generate or load training data
    X, y, feature_extractor = generate_training_data_fast(
        feature_extractor=feature_extractor,
        device=device,
        data_cache_path=data_cache_path,
        max_examples=500000
    )

    # Train model
    model = train_model(X, y, model_save_path, device)

    return model, feature_extractor


if __name__ == "__main__":
    main()
