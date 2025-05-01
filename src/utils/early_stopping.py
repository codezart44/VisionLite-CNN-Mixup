
class EarlyStopper:
    """ Early Stopper """
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def check_stop(self, test_loss: float) -> bool:
        """..."""
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
    
    def reset_stopper(self) -> None:
        """..."""
        self.best_loss = float('inf')
        self.counter = 0
