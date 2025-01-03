import torch.nn as nn

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


class EnhancedHangmanNN(nn.Module):
    def __init__(self, input_size=FEATURE_SIZE, hidden_sizes=None):
        super(EnhancedHangmanNN, self).__init__()

        # Main network
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
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
