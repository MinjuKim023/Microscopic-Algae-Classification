import torch
import os

class EarlyStopping():
    '''
    Copied from my previous paper (https://github.com/babaling/DRPreter/blob/main/utils.py) and slightly modified
    '''
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):

        if metric is not None:
            if metric in ['accuracy']:
                print(f'For metric {metric}, the higher the better')
                mode = 'higher'
            if metric in ['loss']:
                print(f'For metric {metric}, the lower the better')
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """
        Check if the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """
        Check if the new score is lower than the previous best score.
        """ 
        return score < prev_best_score

    def step(self, score, model):
        """
        Update based on a new score.
        The new score is typically model performance on the validation set for a new epoch.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        """
        Load the latest checkpoint
        """        
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

    