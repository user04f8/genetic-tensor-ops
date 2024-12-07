import torch
import torch.nn as nn
from population import PopulationManager

class Trainer:
    def __init__(self, input_shapes, output_shape, population_size=50):
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.population_manager = PopulationManager(input_shapes, output_shape, population_size)
        self.loss_fn = nn.MSELoss()

    def fit(self, train_data, val_data, generations=10, complexity_penalty_factor=0.1):
        for g in range(generations):
            self.population_manager.evolve(train_data, val_data, self.loss_fn, complexity_penalty_factor=complexity_penalty_factor)
            best, best_loss = self.population_manager.best_candidate(train_data, val_data, self.loss_fn)
            print(f"Generation {g}: best_loss={best_loss}, best_model={best}")

        best, best_loss = self.population_manager.best_candidate(train_data, val_data, self.loss_fn)
        return best, best_loss

    def predict(self, model, *Xs):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        Xs = tuple(x.to(device) for x in Xs)
        with torch.no_grad():
            return model(*Xs).cpu()
