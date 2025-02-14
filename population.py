from concurrent.futures import ProcessPoolExecutor
from typing import Iterable
import torch
import random

from gm import GeneticModule
from module_factory import ModuleFactory

class PopulationManager:
    def __init__(self, input_shapes, output_shape, population_size=50, module_factory=None):
        # ensure these are simple tuples
        input_shapes = tuple(tuple(int(s) for s in shape) for shape in input_shapes)
        output_shape = tuple(int(s) for s in output_shape)

        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.population_size = population_size
        self.module_factory = module_factory if module_factory else ModuleFactory()
        self.population = [self.module_factory.create_random_module(input_shapes, output_shape) 
                           for _ in range(population_size)]

    def evaluate_model(self, model, val_Xs, val_Y, loss_fn, device):
        model.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(*val_Xs)
            loss = loss_fn(pred, val_Y).item()
            complexity = model.compute_complexity()
        return loss, complexity

    def evaluate(
        self,
        candidates: Iterable[GeneticModule],
        train_data,
        val_data: tuple[tuple[torch.Tensor, ...], torch.Tensor],
        loss_fn
    ) -> tuple[torch.Tensor, torch.Tensor, GeneticModule]:
        val_Xs, val_Y = val_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move all data to GPU
        val_Xs = tuple(x.to(device) for x in val_Xs)
        val_Y = val_Y.to(device)

        # Move all models to GPU
        for model in candidates:
            model.to(device)
            model.eval()

        # Batched evaluation
        losses = []
        complexities = []
        with torch.no_grad():
            for model in candidates:
                pred = model(*val_Xs)  # Forward pass
                loss = loss_fn(pred, val_Y).item()  # Compute loss
                complexity = model.compute_complexity()  # Compute complexity
                losses.append(loss)
                complexities.append(complexity)

        # Convert to tensors
        losses_tensor = torch.tensor(losses, device=device)
        complexities_tensor = torch.tensor(complexities, device=device)

        # Find the best model
        best_idx = torch.argmin(losses_tensor).item()
        best_model = candidates[best_idx]

        return losses_tensor, complexities_tensor, best_model

    def train_parameters(self, candidate, train_data, loss_fn, lr=0.01, steps=50):
        train_Xs, train_Y = train_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        candidate.to(device)
        candidate.parameters_requires_grad(True)
        params = list(candidate.parameters())
        if len(params) == 0:
            return
        optimizer = torch.optim.Adam(params, lr=lr)

        train_Xs = tuple(x.to(device) for x in train_Xs)
        train_Y = train_Y.to(device)

        candidate.train()
        for _ in range(steps):
            optimizer.zero_grad()
            pred = candidate(*train_Xs)
            loss = loss_fn(pred, train_Y)
            if not loss.requires_grad:
                break
            loss.backward()
            optimizer.step()

        candidate.parameters_requires_grad(False)
        candidate.eval()

    def evolve(self, train_data, val_data, loss_fn, elite_frac=0.2, mutation_rate=0.5, complexity_penalty_factor=0.1):
        print("Setting up optim")
        streams = [torch.cuda.Stream() for _ in range(len(self.population))]

        for i, candidate in enumerate(self.population):
            if any(p.requires_grad for p in candidate.parameters()):
                with torch.cuda.stream(streams[i]):
                    self.train_parameters(candidate, train_data, loss_fn)

        print("Awaiting train")
        torch.cuda.synchronize()
        
        print("Evaluating")
        losses, complexities, best_model = self.evaluate(self.population, train_data, val_data, loss_fn)
        fitness = losses + complexity_penalty_factor*complexities

        print(f'Means: loss={losses.mean()}, complexity={complexities.mean()} \n best<{losses.min()}> = {best_model}')

        print("Determining survivors")
        ranked = sorted(zip(self.population, fitness), key=lambda x: x[1])
        elites_count = max(1, int(self.population_size * elite_frac))
        survivors = [p for p, l in ranked[:elites_count]]

        # Decide who else survives
        for i, (p, l) in enumerate(ranked[elites_count:]):
            p_kill = i / self.population_size
            if random.random() > p_kill:
                survivors.append(p)

        self.population = survivors[:self.population_size]

        print("Mutating")
        for i in range(elites_count, len(survivors)):
            if random.random() < mutation_rate:
                mutated = survivors[i].mutate(self.module_factory)
                if mutated:
                    survivors[i] = mutated

        print(f"Filling with {self.population_size-len(survivors)} new models")
        for i in range(len(survivors)-self.population_size):
            # new_mod = self.module_factory.create_random_module(self.input_shapes, self.output_shape)
            survivors.append(ranked[i].copy().mutate())
    
    def best_candidate(self, train_data, val_data, loss_fn):
        best_loss, _, best_model = self.evaluate(self.population, train_data, val_data, loss_fn)
        return best_model, best_loss
