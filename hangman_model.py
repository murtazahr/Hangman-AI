import torch.nn as nn

MAX_WORD_LENGTH = 20
NUM_LETTERS = 26
NUM_ADDITIONAL_FEATURES = 2  # word length and num guesses
FEATURE_SIZE = MAX_WORD_LENGTH + NUM_LETTERS + NUM_ADDITIONAL_FEATURES


class HangmanNN(nn.Module):
    def __init__(self, input_size=FEATURE_SIZE, hidden_size=128):
        super(HangmanNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 26)  # Output for each letter probability
        )

    def forward(self, x):
        return self.network(x)
