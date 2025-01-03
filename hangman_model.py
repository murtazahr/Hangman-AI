import string

import torch.nn as nn

# Constants
MAX_WORD_LENGTH = 20
NUM_LETTERS = len(string.ascii_lowercase)  # 26

# Calculate feature size components
PATTERN_FEATURES = MAX_WORD_LENGTH  # 20
GUESSED_FEATURES = NUM_LETTERS  # 26
NGRAM_FEATURES = NUM_LETTERS * 3  # 78 (unigram, bigram, and trigram scores)
LETTER_FREQ_FEATURES = NUM_LETTERS  # 26
STRUCTURE_FEATURES = 2  # 2
RATIO_FEATURES = 2  # 2
REMAINING_FEATURES = NUM_LETTERS  # 26 (scores for remaining positions)
REMAINING_META_FEATURES = 3  # 3 (num_remaining, remaining_guesses, context_importance)

# Total feature size
FEATURE_SIZE = (
        PATTERN_FEATURES +  # 20
        GUESSED_FEATURES +  # 26
        NGRAM_FEATURES +  # 78
        LETTER_FREQ_FEATURES +  # 26
        STRUCTURE_FEATURES +  # 2
        RATIO_FEATURES +  # 2
        REMAINING_FEATURES +  # 26
        REMAINING_META_FEATURES  # 3
)  # Total: 183


class HangmanNN(nn.Module):
    def __init__(self, input_size=FEATURE_SIZE, hidden_sizes=None):
        super(HangmanNN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]  # Deeper network for more complex features

        print(f"Initializing model with input size: {input_size}")

        # Build network layers
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3 if i == len(hidden_sizes)-1 else 0.2)  # Higher dropout in final layer
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, NUM_LETTERS))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
