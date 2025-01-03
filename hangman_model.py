import string

import torch.nn as nn

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


class HangmanNN(nn.Module):
    def __init__(self, input_size=FEATURE_SIZE, hidden_sizes=None):
        super(HangmanNN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        # Build network layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, NUM_LETTERS))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
