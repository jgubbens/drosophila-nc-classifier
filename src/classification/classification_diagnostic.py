import torch

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
    viterbi_hmm = HMM_Trainer(DATA_PATH, DEVICE, window_size=5, preprocess_images=False, lstm_module=False, img_size=(800, 800))
    viterbi_hmm.train_hmm()
