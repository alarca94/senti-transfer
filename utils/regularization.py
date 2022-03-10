class EarlyStopping:
    def __init__(self, patience, delta=0, path='./models/pretrained'):
        self.patience = patience    # Maximum number of epochs to wait for a validation loss improvement
        self.path = path
        self.delta = delta          # Minimum improvement to be qualified as such

        self.patience_counter = 0
        self.checkpoint_epoch = 0
        self.early_stop = False
        self.best_val_scores = None
        self.best_trn_scores = None
        self.best_model_name = ''

    def __call__(self, trn_scores, val_scores, model, save=True):
        if self.best_val_scores is None or self.best_val_scores['loss'] > (val_scores['loss'] + self.delta):
            self.checkpoint_epoch += self.patience_counter + 1
            self.patience_counter = 0
            self.best_val_scores = val_scores.copy()
            self.best_trn_scores = trn_scores.copy()
            self.best_model_name = model.name
            if save:
                model.save(self.path)
        else:
            self.patience_counter += 1
            print(f'Patience is running out... {self.patience - self.patience_counter} steps remaining...')
            if self.patience_counter >= self.patience:
                print(f'\nLast checkpoint occurred at epoch {self.checkpoint_epoch} with validation loss '
                      f'{self.best_val_scores["loss"]:.4f}')
                self.early_stop = True

    def reset(self):
        self.patience_counter = 0
        self.checkpoint_epoch = 0
        self.early_stop = False
        self.best_trn_scores = None
        self.best_val_scores = None
        self.best_model_name = ''
