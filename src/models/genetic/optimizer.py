"""
Genetic Algorithm Optimizer for trading strategy parameters.
"""

import numpy as np
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from loguru import logger


@dataclass
class Individual:
    """Represents an individual in the population."""
    genes: Dict[str, float]
    fitness: float = 0.0
    generation: int = 0

    def to_dict(self) -> Dict:
        return {
            'genes': self.genes,
            'fitness': self.fitness,
            'generation': self.generation
        }


@dataclass
class ParameterRange:
    """Defines the range for a parameter."""
    name: str
    min_value: float
    max_value: float
    is_integer: bool = False
    step: Optional[float] = None


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for trading strategy parameters.
    Optimizes parameters like take profit, stop loss, indicator periods, etc.
    """

    def __init__(
        self,
        parameter_ranges: List[ParameterRange],
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 5,
        elite_size: int = 10,
        random_seed: Optional[int] = None
    ):
        """
        Initialize genetic optimizer.

        Args:
            parameter_ranges: List of parameter ranges to optimize
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament selection
            elite_size: Number of elite individuals to preserve
            random_seed: Random seed for reproducibility
        """
        self.parameter_ranges = {p.name: p for p in parameter_ranges}
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []

        logger.info(f"GeneticOptimizer initialized with {len(parameter_ranges)} parameters")

    def _create_random_genes(self) -> Dict[str, float]:
        """Create random genes within parameter ranges."""
        genes = {}
        for name, param in self.parameter_ranges.items():
            if param.is_integer:
                value = random.randint(int(param.min_value), int(param.max_value))
            elif param.step:
                steps = int((param.max_value - param.min_value) / param.step)
                value = param.min_value + random.randint(0, steps) * param.step
            else:
                value = random.uniform(param.min_value, param.max_value)
            genes[name] = value
        return genes

    def _initialize_population(self) -> None:
        """Initialize random population."""
        self.population = [
            Individual(genes=self._create_random_genes())
            for _ in range(self.population_size)
        ]

    def _tournament_selection(self) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        # Uniform crossover
        child1_genes = {}
        child2_genes = {}

        for name in self.parameter_ranges:
            if random.random() < 0.5:
                child1_genes[name] = parent1.genes[name]
                child2_genes[name] = parent2.genes[name]
            else:
                child1_genes[name] = parent2.genes[name]
                child2_genes[name] = parent1.genes[name]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _blend_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        alpha: float = 0.5
    ) -> Tuple[Individual, Individual]:
        """Perform BLX-alpha crossover for continuous optimization."""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        child1_genes = {}
        child2_genes = {}

        for name, param in self.parameter_ranges.items():
            p1_val = parent1.genes[name]
            p2_val = parent2.genes[name]

            min_val = min(p1_val, p2_val)
            max_val = max(p1_val, p2_val)
            diff = max_val - min_val

            lower = max(param.min_value, min_val - alpha * diff)
            upper = min(param.max_value, max_val + alpha * diff)

            c1_val = random.uniform(lower, upper)
            c2_val = random.uniform(lower, upper)

            if param.is_integer:
                c1_val = round(c1_val)
                c2_val = round(c2_val)

            child1_genes[name] = c1_val
            child2_genes[name] = c2_val

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        new_genes = individual.genes.copy()

        for name, param in self.parameter_ranges.items():
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                std = (param.max_value - param.min_value) * 0.1
                new_value = new_genes[name] + random.gauss(0, std)

                # Clamp to range
                new_value = max(param.min_value, min(param.max_value, new_value))

                if param.is_integer:
                    new_value = round(new_value)

                new_genes[name] = new_value

        return Individual(genes=new_genes)

    def _evaluate_population(
        self,
        fitness_function: Callable[[Dict[str, float]], float]
    ) -> None:
        """Evaluate fitness of all individuals."""
        for individual in self.population:
            if individual.fitness == 0.0:  # Not yet evaluated
                individual.fitness = fitness_function(individual.genes)

    def _evolve_population(self) -> List[Individual]:
        """Create next generation."""
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # Elite selection
        new_population = sorted_pop[:self.elite_size]

        # Generate rest through selection and crossover
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            child1, child2 = self._blend_crossover(parent1, parent2)

            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            new_population.extend([child1, child2])

        return new_population[:self.population_size]

    def optimize(
        self,
        fitness_function: Callable[[Dict[str, float]], float],
        early_stopping: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Run genetic algorithm optimization.

        Args:
            fitness_function: Function that evaluates parameter fitness
            early_stopping: Stop if no improvement for this many generations
            verbose: Print progress

        Returns:
            Best parameters found
        """
        logger.info(f"Starting optimization for {self.generations} generations")

        # Initialize population
        self._initialize_population()

        best_fitness = float('-inf')
        no_improvement = 0

        for generation in range(self.generations):
            # Evaluate fitness
            self._evaluate_population(fitness_function)

            # Update generation number
            for ind in self.population:
                ind.generation = generation

            # Track best
            current_best = max(self.population, key=lambda x: x.fitness)

            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                self.best_individual = current_best
                no_improvement = 0
            else:
                no_improvement += 1

            # Record history
            fitnesses = [ind.fitness for ind in self.population]
            self.history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses),
                'best_genes': self.best_individual.genes.copy()
            })

            if verbose and (generation + 1) % 5 == 0:
                logger.info(
                    f"Generation {generation + 1}/{self.generations} - "
                    f"Best: {best_fitness:.6f} - Mean: {np.mean(fitnesses):.6f}"
                )

            # Early stopping
            if no_improvement >= early_stopping:
                logger.info(f"Early stopping at generation {generation + 1}")
                break

            # Evolve
            self.population = self._evolve_population()

        logger.info(f"Optimization complete. Best fitness: {best_fitness:.6f}")

        return {
            'best_parameters': self.best_individual.genes,
            'best_fitness': self.best_individual.fitness,
            'generations_run': generation + 1,
            'history': self.history
        }

    def optimize_strategy(
        self,
        backtest_function: Callable[[Dict[str, float]], Dict],
        metric: str = 'sharpe_ratio',
        minimize: bool = False
    ) -> Dict:
        """
        Optimize trading strategy parameters.

        Args:
            backtest_function: Function that runs backtest and returns metrics
            metric: Metric to optimize (sharpe_ratio, profit_factor, etc.)
            minimize: Whether to minimize the metric

        Returns:
            Optimized parameters and results
        """
        def fitness_wrapper(params: Dict[str, float]) -> float:
            try:
                results = backtest_function(params)
                value = results.get(metric, 0)

                # Handle special cases
                if np.isnan(value) or np.isinf(value):
                    return float('-inf') if not minimize else float('inf')

                return -value if minimize else value
            except Exception as e:
                logger.error(f"Backtest error: {e}")
                return float('-inf') if not minimize else float('inf')

        return self.optimize(fitness_wrapper)

    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Analyze parameter importance based on fitness correlation.

        Returns:
            Dictionary of parameter importance scores
        """
        if len(self.history) < 5:
            return {name: 1.0 / len(self.parameter_ranges) for name in self.parameter_ranges}

        # Collect all evaluated individuals
        all_genes = []
        all_fitness = []

        for record in self.history:
            all_genes.append(record['best_genes'])
            all_fitness.append(record['best_fitness'])

        # Calculate correlation with fitness for each parameter
        importance = {}
        for name in self.parameter_ranges:
            param_values = [g[name] for g in all_genes]
            correlation = np.corrcoef(param_values, all_fitness)[0, 1]
            importance[name] = abs(correlation) if not np.isnan(correlation) else 0.0

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def save(self, filepath: Union[str, Path]) -> None:
        """Save optimizer state."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'parameter_ranges': {
                name: {
                    'min_value': p.min_value,
                    'max_value': p.max_value,
                    'is_integer': p.is_integer,
                    'step': p.step
                }
                for name, p in self.parameter_ranges.items()
            },
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'history': self.history,
            'config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Optimizer state saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """Load optimizer state."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        if data.get('best_individual'):
            self.best_individual = Individual(**data['best_individual'])

        self.history = data.get('history', [])
        logger.info(f"Optimizer state loaded from {filepath}")


# Predefined parameter ranges for common trading strategies
DEFAULT_PARAMETER_RANGES = [
    ParameterRange('take_profit', 0.01, 0.10, step=0.005),
    ParameterRange('stop_loss', 0.01, 0.05, step=0.005),
    ParameterRange('trailing_stop', 0.005, 0.03, step=0.005),
    ParameterRange('rsi_period', 7, 21, is_integer=True),
    ParameterRange('rsi_overbought', 65, 85, is_integer=True),
    ParameterRange('rsi_oversold', 15, 35, is_integer=True),
    ParameterRange('ma_fast', 5, 20, is_integer=True),
    ParameterRange('ma_slow', 20, 100, is_integer=True),
    ParameterRange('atr_multiplier', 1.0, 3.0, step=0.25),
    ParameterRange('position_size', 0.1, 0.5, step=0.05),
]
