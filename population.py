from typing import Iterable
import torch
import random
import math
from copy import deepcopy

from genetic_module import GeneticModule
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

    def evaluate(self, candidates: Iterable[GeneticModule], train_data, val_data, loss_fn, complexity_penalty_factor):
        val_Xs, val_Y = val_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_Xs = tuple(x.to(device) for x in val_Xs)
        val_Y = val_Y.to(device)
        losses = []
        for c in candidates:
            c.to(device)
            c.eval()
            with torch.no_grad():
                pred = c(*val_Xs)
                l = loss_fn(pred, val_Y) + complexity_penalty_factor*c.complexity()
            losses.append(l.item())
        return losses

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
                # No point in calling backward if no gradient can flow
                break
            
            loss.backward()
            optimizer.step()

        candidate.parameters_requires_grad(False)
        candidate.eval()

    def evolve(self, train_data, val_data, loss_fn, elite_frac=0.2, mutation_rate=0.5, complexity_penalty_factor=0.1):
        # Train parameters
        for candidate in self.population:
            if any(p.requires_grad for p in candidate.parameters()):
                self.train_parameters(candidate, train_data, loss_fn)

        # Evaluate
        losses = self.evaluate(self.population, train_data, val_data, loss_fn, complexity_penalty_factor)
        mean_loss = sum(losses)/len(losses)

        # Kill-off probability
        # Probability of killing a candidate i: p_kill(i) ~ (loss_i / mean_loss)
        # Then we keep or kill each candidate except elites.
        ranked = sorted(zip(self.population, losses), key=lambda x: x[1])
        elites_count = max(1, int(self.population_size * elite_frac))
        survivors = [p for p, l in ranked[:elites_count]]

        # Decide who else survives
        for p, l in ranked[elites_count:]:
            p_kill = l/mean_loss
            if random.random() > p_kill:
                survivors.append(p)

        # Refill population if needed
        while len(survivors) < self.population_size:
            new_mod = self.module_factory.create_random_module(self.input_shapes, self.output_shape)
            if new_mod is not None:
                survivors.append(new_mod)

        # Mutate some survivors to maintain diversity
        for i in range(elites_count, len(survivors)):
            if random.random() < mutation_rate:
                mutated = survivors[i].mutate(self.module_factory)
                if mutated:
                    survivors[i] = mutated

        # if still less than population_size, fill with random
        while len(survivors) < self.population_size:
            new_mod = self.module_factory.create_random_module(self.input_shapes, self.output_shape)
            if new_mod:
                survivors.append(new_mod)

        self.population = survivors[:self.population_size]

    def best_candidate(self, train_data, val_data, loss_fn):
        losses = self.evaluate(self.population, train_data, val_data, loss_fn, 0)
        best_idx = min(range(len(losses)), key=lambda i: losses[i])
        return self.population[best_idx], losses[best_idx]
