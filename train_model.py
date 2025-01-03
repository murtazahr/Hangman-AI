import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
import re
from collections import Counter
from tqdm.auto import tqdm

from hangman_model import EnhancedHangmanNN

# Constants
MAX_WORD_LENGTH = 20
NUM_LETTERS = 26
NGRAM_SIZE = 3
FEATURE_SIZE = (
        MAX_WORD_LENGTH +  # One-hot word pattern
        NUM_LETTERS +  # Letter presence
        NUM_LETTERS * 2 +  # Position-specific letter probabilities (start/end)
        NUM_LETTERS +  # Letter frequencies in matching words
        NGRAM_SIZE * NUM_LETTERS +  # N-gram features
        4  # Additional features (word length, num guesses, vowel ratio, consonant ratio)
)


class EnhancedFeatureExtractor:
    def __init__(self, dictionary):
        print("Initializing Enhanced Feature Extractor...")

        self.dictionary = dictionary
        print(f"Dictionary size: {len(dictionary)} words")

        print("Building frequency tables...")
        with tqdm(total=2, desc="Building frequency data") as pbar:
            self.letter_pos_freq = self._build_position_frequencies()
            pbar.update(1)
            self.ngram_freq = self._build_ngram_frequencies()
            pbar.update(1)

    def _build_position_frequencies(self):
        """Build letter frequency dictionary for start and end positions"""
        print("Building position frequencies...")
        start_freq = Counter()
        end_freq = Counter()

        for word in tqdm(self.dictionary, desc="Analyzing word positions"):
            if word:
                start_freq[word[0]] += 1
                end_freq[word[-1]] += 1

        # Normalize frequencies
        total_words = len(self.dictionary)
        start_probs = {letter: count / total_words for letter, count in start_freq.items()}
        end_probs = {letter: count / total_words for letter, count in end_freq.items()}

        return {'start': start_probs, 'end': end_probs}

    def _build_ngram_frequencies(self):
        """Build n-gram frequency dictionary"""
        print("Building n-gram frequencies...")
        ngram_freq = [Counter() for _ in range(NGRAM_SIZE)]
        normalized_freq = []

        # Count frequencies
        for word in tqdm(self.dictionary, desc="Analyzing n-grams"):
            for n in range(1, NGRAM_SIZE + 1):
                for i in range(len(word) - n + 1):
                    ngram = word[i:i + n]
                    ngram_freq[n - 1][ngram] += 1

        # Normalize frequencies
        print("Normalizing n-gram frequencies...")
        for counter in tqdm(ngram_freq, desc="Normalizing"):
            total = sum(counter.values())
            if total > 0:
                normalized = Counter({k: v / total for k, v in counter.items()})
            else:
                normalized = Counter()
            normalized_freq.append(normalized)

        return normalized_freq

    def get_matching_words(self, pattern, guessed_letters):
        """Get all words matching the current pattern"""
        # Handle both string and list patterns
        if isinstance(pattern, list):
            clean_pattern = "".join(pattern)
        else:
            clean_pattern = pattern[::2]  # Remove spaces if string input

        # Convert to regex pattern
        regex_pattern = clean_pattern.replace("_", ".")
        len_word = len(clean_pattern)

        # Find matching words using regex
        matching_words = [
            word for word in self.dictionary
            if len(word) == len_word and re.match(regex_pattern, word)
        ]

        return matching_words

    def extract_features(self, word_pattern, guessed_letters, device):
        """Extract enhanced feature vector"""
        features = []

        # Clean pattern handling both string and list inputs
        if isinstance(word_pattern, list):
            clean_pattern = "".join(word_pattern)
        else:
            clean_pattern = word_pattern[::2]  # Remove spaces if string input
        pattern_len = len(clean_pattern)

        # Get matching words for this pattern
        matching_words = self.get_matching_words(word_pattern, guessed_letters)

        # Build features in sequence
        feature_components = {
            'pattern': self._get_pattern_features(clean_pattern),
            'letters': self._get_letter_features(guessed_letters),
            'position': self._get_position_features(matching_words),
            'frequency': self._get_frequency_features(matching_words),
            'ngram': self._get_ngram_features(matching_words),
            'metadata': self._get_metadata_features(clean_pattern, guessed_letters, matching_words)
        }

        # Combine all features
        for component in feature_components.values():
            features.extend(component)

        assert len(features) == FEATURE_SIZE, f"Feature size mismatch: {len(features)} != {FEATURE_SIZE}"
        return torch.FloatTensor(features).to(device)

    def _get_pattern_features(self, clean_pattern):
        """Extract pattern-based features"""
        pattern_features = []
        for pos in range(MAX_WORD_LENGTH):
            if pos < len(clean_pattern):
                pattern_features.append(1 if clean_pattern[pos] != '_' else 0)
            else:
                pattern_features.append(0)
        return pattern_features

    def _get_letter_features(self, guessed_letters):
        """Extract letter-based features"""
        return [1 if letter in guessed_letters else 0 for letter in string.ascii_lowercase]

    def _get_position_features(self, matching_words):
        """Extract position-specific features"""
        features = []

        # Start position probabilities
        for letter in string.ascii_lowercase:
            features.append(self.letter_pos_freq['start'].get(letter, 0))

        # End position probabilities
        for letter in string.ascii_lowercase:
            features.append(self.letter_pos_freq['end'].get(letter, 0))

        return features

    def _get_frequency_features(self, matching_words):
        """Extract frequency-based features"""
        if matching_words:
            freq = Counter("".join(matching_words))
            total = sum(freq.values())
            return [freq.get(letter, 0) / total if total > 0 else 0
                    for letter in string.ascii_lowercase]
        return [0] * NUM_LETTERS

    def _get_ngram_features(self, matching_words):
        """Extract n-gram based features"""
        features = []

        for n in range(NGRAM_SIZE):
            letter_scores = {letter: 0.0 for letter in string.ascii_lowercase}

            if matching_words:
                for word in matching_words:
                    for i in range(len(word) - n):
                        ngram = word[i:i + n + 1]
                        if ngram in self.ngram_freq[n]:
                            for letter in ngram:
                                letter_scores[letter] += self.ngram_freq[n][ngram]

            features.extend(letter_scores.values())

        return features

    def _get_metadata_features(self, clean_pattern, guessed_letters, matching_words):
        """Extract metadata features"""
        features = []

        # Word length (normalized)
        features.append(len(clean_pattern) / MAX_WORD_LENGTH)

        # Number of guesses (normalized)
        features.append(len(guessed_letters) / 6.0)

        # Vowel and consonant ratios
        remaining_positions = clean_pattern.count('_')
        if remaining_positions > 0:
            vowel_ratio = sum(1 for pos, char in enumerate(clean_pattern)
                              if char == '_' and any(v in self.get_likely_letters(pos, matching_words)
                                                     for v in 'aeiou')) / remaining_positions
            consonant_ratio = 1 - vowel_ratio
        else:
            vowel_ratio = consonant_ratio = 0

        features.extend([vowel_ratio, consonant_ratio])

        return features

    def get_likely_letters(self, position, matching_words):
        """Get likely letters for a position based on matching words"""
        letters = set()
        for word in matching_words:
            if position < len(word):
                letters.add(word[position])
        return letters


def create_enhanced_model(dictionary_path, model_save_path, data_cache_path="data/training_data.pt"):
    """Create and train enhanced model with validation, using cached data if available"""
    from pathlib import Path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for cached training data
    data_cache_file = Path(data_cache_path)
    if data_cache_file.exists():
        print("Loading cached training data...")
        cached_data = torch.load(data_cache_file, map_location=device)
        X_train = cached_data['X_train']
        y_train = cached_data['y_train']
        X_val = cached_data['X_val']
        y_val = cached_data['y_val']
        feature_extractor = cached_data['feature_extractor']

        print(f"Loaded cached data - Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    else:
        print("No cached data found. Generating training data...")
        # Load dictionary and create feature extractor
        with open(dictionary_path, 'r') as f:
            dictionary = f.read().splitlines()

        feature_extractor = EnhancedFeatureExtractor(dictionary)

        # Generate training data
        print("Generating training data...")
        X, y = [], []

        # Filter valid words first
        valid_words = [word for word in dictionary if len(word) <= MAX_WORD_LENGTH]

        for word in tqdm(valid_words, desc="Generating training examples"):
            # Generate multiple examples per word
            for _ in range(5):
                guessed = []
                pattern = ['_'] * len(word)

                # Simulate random game states
                num_guesses = np.random.randint(0, 5)
                for _ in range(num_guesses):
                    letter = np.random.choice(list(string.ascii_lowercase))
                    if letter not in guessed:
                        guessed.append(letter)
                        for i, char in enumerate(word):
                            if char == letter:
                                pattern[i] = letter

                features = feature_extractor.extract_features(pattern, guessed, device)

                # Create target vector with weighted probabilities
                target = np.zeros(26)
                remaining_letters = set(word) - set(guessed)
                if remaining_letters:
                    # Weight letters by their position in the word
                    total_weight = 0
                    for letter in remaining_letters:
                        weight = 1.0
                        if letter in word[0]:  # First letter bonus
                            weight *= 1.5
                        if letter in word[-1]:  # Last letter bonus
                            weight *= 1.5
                        if letter in 'aeiou':  # Vowel bonus
                            weight *= 1.2
                        target[ord(letter) - ord('a')] = weight
                        total_weight += weight

                    # Normalize weights
                    if total_weight > 0:
                        target = target / total_weight

                X.append(features)
                y.append(torch.FloatTensor(target).to(device))

        X = torch.stack(X)
        y = torch.stack(y)

        print(f"Generated {len(X)} training examples")

        # Split into train/validation sets
        indices = torch.randperm(len(X))
        split = int(0.9 * len(X))
        train_indices = indices[:split]
        val_indices = indices[split:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Save the processed data
        print("Saving training data to cache...")
        data_cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_extractor': feature_extractor
        }, data_cache_file)
        print(f"Data cached to {data_cache_file}")

    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # Create and train model
    model = EnhancedHangmanNN().to(device)
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
                             desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for i in train_batches:
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

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
                           desc=f"Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for i in val_batches:
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                val_batches.set_postfix({'loss': loss.item()})

        avg_val_loss = total_val_loss / len(val_batches)

        print(f"\nEpoch {epoch+1}/{epochs}")
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
    return feature_extractor


if __name__ == '__main__':
    feature_extractor = create_enhanced_model(
        dictionary_path="words_250000_train.txt",
        model_save_path="models/enhanced_hangman_model.pt"
    )
