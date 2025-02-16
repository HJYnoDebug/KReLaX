class Hyperparameters:
    def __init__(self):
        self.learning_rate = 1e-5
        self.batch_size = 16
        self.num_epochs = 15
        self.max_len = 128
        self.early_stopping_patience = 2
        self.weight_decay = 0.01

    def hyper_search(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            return self


