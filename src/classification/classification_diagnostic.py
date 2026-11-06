from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.greedy_hmm.greedy_hmm_trainer import Greedy_HMM
from classification.viterbi_hmm.viterbi_hmm_trainer import Viterbi_HMM

if __name__ == '__main__':

    DATA_PATH = r'data/training_data'

    print('Running hidden markov model classification benchmarks')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')

    print("Greedy HMM:")
    greedy_hmm = Greedy_HMM(DATA_PATH, DEVICE, window_size=5, preprocess_images=False, lstm_module=False, img_size=(800, 800))
    greedy_hmm.train_hmm()

    print("Viterbi HMM:")
    # Reuse the CNN and train/val split from the greedy run so only the decoder differs
    viterbi_hmm = Viterbi_HMM(DATA_PATH, DEVICE, window_size=5, preprocess_images=False, lstm_module=False, img_size=(800, 800))
    viterbi_hmm.cnn = greedy_hmm.cnn
    viterbi_hmm.train_vids = greedy_hmm.train_vids
    viterbi_hmm.val_vids = greedy_hmm.val_vids
    Path(viterbi_hmm.model_path).mkdir(parents=True, exist_ok=True)
    viterbi_hmm._additional_training()
    viterbi_hmm.save_model_info()
    viterbi_hmm.evaluate()
