"""
Genetic Algorithm Engine for Portfolio Optimization

Implements a complete GA with:
- Real-valued chromosome encoding (portfolio weights)
- Sharpe Ratio + MaxDrawdown fitness function
- Tournament selection
- BLX-α crossover (blend crossover)
- Gaussian mutation
- Weight normalization with constraint enforcement
- Elitism preservation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from config import (
    POPULATION_SIZE, NUM_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE,
    MUTATION_SCALE, TOURNAMENT_SIZE, ELITISM_COUNT,
    MAX_WEIGHT, MIN_WEIGHT, RISK_FREE_RATE,
    DRAWDOWN_PENALTY, CONSTRAINT_PENALTY
)


@dataclass
class Individual:
    """Represents a single portfolio allocation (chromosome)."""
    weights: np.ndarray
    fitness: float = 0.0
    sharpe: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class GAResult:
    """Result of a GA optimization run."""
    best_individual: Individual
    best_fitness_history: List[float] = field(default_factory=list)
    avg_fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    final_population: List[Individual] = field(default_factory=list)
    generations_run: int = 0


class GeneticAlgorithm:
    """
    Genetic Algorithm optimizer for crypto portfolio weights.
    
    Architecture:
        Binance API → Data Collector → [GA Optimizer] → Backtester → Dashboard
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        n_assets: int,
        population_size: int = POPULATION_SIZE,
        num_generations: int = NUM_GENERATIONS,
        crossover_rate: float = CROSSOVER_RATE,
        mutation_rate: float = MUTATION_RATE,
        mutation_scale: float = MUTATION_SCALE,
        tournament_size: int = TOURNAMENT_SIZE,
        elitism_count: int = ELITISM_COUNT,
        max_weight: float = MAX_WEIGHT,
        min_weight: float = MIN_WEIGHT,
        risk_free_rate: float = RISK_FREE_RATE,
        callback: Optional[Callable] = None
    ):
        self.returns = returns
        self.n_assets = n_assets
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free_rate = risk_free_rate
        self.callback = callback
        
        # Precompute statistics
        self.mean_returns = np.mean(returns, axis=0)
        self.cov_matrix = np.cov(returns.T)
        
    # ══════════════════════════════════════════
    # 1. INITIALIZATION
    # ══════════════════════════════════════════
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual with valid weights."""
        weights = np.random.dirichlet(np.ones(self.n_assets))
        weights = self._enforce_constraints(weights)
        return Individual(weights=weights)
    
    def _initialize_population(self) -> List[Individual]:
        """Create initial population of random portfolios."""
        population = []
        for _ in range(self.population_size):
            ind = self._create_random_individual()
            population.append(ind)
        return population
    
    # ══════════════════════════════════════════
    # 2. FITNESS FUNCTION
    # ══════════════════════════════════════════
    
    def _compute_portfolio_returns(self, weights: np.ndarray) -> np.ndarray:
        """Compute daily portfolio returns given weights."""
        return self.returns @ weights
    
    def _compute_sharpe_ratio(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute annualized Sharpe Ratio.
        
        fitness = (Return - Risk_free) / Volatility
        
        Returns:
            (sharpe_ratio, annual_return, annual_volatility)
        """
        portfolio_return = np.sum(self.mean_returns * weights) * 365
        portfolio_volatility = np.sqrt(weights @ self.cov_matrix @ weights) * np.sqrt(365)
        
        if portfolio_volatility < 1e-10:
            return 0.0, portfolio_return, portfolio_volatility
        
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return sharpe, portfolio_return, portfolio_volatility
    
    def _compute_max_drawdown(self, weights: np.ndarray) -> float:
        """Compute maximum drawdown for the portfolio."""
        portfolio_returns = self._compute_portfolio_returns(weights)
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    def _compute_constraint_penalty(self, weights: np.ndarray) -> float:
        """
        Compute penalty for constraint violations.
        
        Constraints:
        - sum(weights) = 1
        - weight_i <= MAX_WEIGHT (prevent all-in)
        - weight_i >= MIN_WEIGHT (prevent dust allocations)
        """
        penalty = 0.0
        
        # Sum constraint (should already be enforced but just in case)
        sum_violation = abs(np.sum(weights) - 1.0)
        penalty += sum_violation * CONSTRAINT_PENALTY
        
        # Max weight constraint
        max_violations = np.maximum(weights - self.max_weight, 0)
        penalty += np.sum(max_violations) * CONSTRAINT_PENALTY
        
        # Min weight constraint
        min_violations = np.maximum(self.min_weight - weights, 0)
        penalty += np.sum(min_violations) * CONSTRAINT_PENALTY
        
        return penalty
    
    def _evaluate_fitness(self, individual: Individual) -> float:
        """
        Advanced Fitness Function:
        
        fitness = Sharpe - 0.5 * MaxDrawdown - penalty(constraints)
        """
        sharpe, ann_ret, ann_vol = self._compute_sharpe_ratio(individual.weights)
        max_dd = self._compute_max_drawdown(individual.weights)
        penalty = self._compute_constraint_penalty(individual.weights)
        
        fitness = sharpe - DRAWDOWN_PENALTY * max_dd - penalty
        
        individual.sharpe = sharpe
        individual.annual_return = ann_ret
        individual.annual_volatility = ann_vol
        individual.max_drawdown = max_dd
        individual.fitness = fitness
        
        return fitness
    
    def _evaluate_population(self, population: List[Individual]):
        """Evaluate fitness for the entire population."""
        for ind in population:
            self._evaluate_fitness(ind)
    
    # ══════════════════════════════════════════
    # 3. SELECTION (Tournament)
    # ══════════════════════════════════════════
    
    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Select an individual using tournament selection."""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament = [population[i] for i in indices]
        winner = max(tournament, key=lambda x: x.fitness)
        return winner
    
    # ══════════════════════════════════════════
    # 4. CROSSOVER (BLX-α Blend)
    # ══════════════════════════════════════════
    
    def _blx_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        BLX-α (Blend) Crossover - good for real-valued chromosomes.
        Creates offspring within an expanded range of parent values.
        """
        if np.random.random() > self.crossover_rate:
            return (
                Individual(weights=parent1.weights.copy()),
                Individual(weights=parent2.weights.copy())
            )
        
        alpha = 0.5
        child1_weights = np.zeros(self.n_assets)
        child2_weights = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            d = abs(parent1.weights[i] - parent2.weights[i])
            low = min(parent1.weights[i], parent2.weights[i]) - alpha * d
            high = max(parent1.weights[i], parent2.weights[i]) + alpha * d
            low = max(low, 0)
            high = max(high, low + 1e-10)
            
            child1_weights[i] = np.random.uniform(low, high)
            child2_weights[i] = np.random.uniform(low, high)
        
        child1_weights = self._enforce_constraints(child1_weights)
        child2_weights = self._enforce_constraints(child2_weights)
        
        return Individual(weights=child1_weights), Individual(weights=child2_weights)
    
    # ══════════════════════════════════════════
    # 5. MUTATION (Gaussian)
    # ══════════════════════════════════════════
    
    def _gaussian_mutate(self, individual: Individual) -> Individual:
        """
        Apply Gaussian mutation to weights.
        Each gene has a probability of being mutated.
        """
        weights = individual.weights.copy()
        
        for i in range(self.n_assets):
            if np.random.random() < self.mutation_rate:
                weights[i] += np.random.normal(0, self.mutation_scale)
                weights[i] = max(weights[i], 0)
        
        weights = self._enforce_constraints(weights)
        individual.weights = weights
        return individual
    
    # ══════════════════════════════════════════
    # 6. NORMALIZE WEIGHTS (สำคัญ!)
    # ══════════════════════════════════════════
    
    def _enforce_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        Enforce portfolio constraints:
        1. All weights >= 0
        2. Each weight within [MIN_WEIGHT, MAX_WEIGHT]
        3. Sum of weights = 1
        """
        # Clip to valid range
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        # Normalize to sum to 1
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(self.n_assets) / self.n_assets
        
        # Re-clip after normalization (may violate bounds)
        # Iterative correction
        for _ in range(10):
            weights = np.clip(weights, self.min_weight, self.max_weight)
            total = np.sum(weights)
            if abs(total - 1.0) < 1e-10:
                break
            # Distribute excess/deficit proportionally among non-bound weights
            diff = 1.0 - total
            free_mask = (weights > self.min_weight + 1e-10) & (weights < self.max_weight - 1e-10)
            free_count = np.sum(free_mask)
            if free_count > 0:
                weights[free_mask] += diff / free_count
            else:
                weights += diff / self.n_assets
        
        # Final normalization
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)
        
        return weights
    
    # ══════════════════════════════════════════
    # 7. POPULATION DIVERSITY
    # ══════════════════════════════════════════
    
    def _compute_diversity(self, population: List[Individual]) -> float:
        """Compute population diversity as average pairwise distance."""
        if len(population) < 2:
            return 0.0
        
        weights_matrix = np.array([ind.weights for ind in population])
        # Use standard deviation of each weight across population
        diversity = np.mean(np.std(weights_matrix, axis=0))
        return diversity
    
    # ══════════════════════════════════════════
    # MAIN GA LOOP
    # ══════════════════════════════════════════
    
    def run(self) -> GAResult:
        """
        Execute the Genetic Algorithm.
        
        GA Workflow:
        1. สร้าง population เริ่มต้น (random weights)
        2. คำนวณ fitness
        3. คัดเลือก (Selection)
        4. ผสมพันธุ์ (Crossover)
        5. กลายพันธุ์ (Mutation)
        6. Normalize weights (สำคัญ!)
        7. ทำซ้ำ 100-500 generations
        """
        result = GAResult(best_individual=None)
        
        # Step 1: Initialize population
        population = self._initialize_population()
        
        # Step 2: Evaluate initial population
        self._evaluate_population(population)
        
        for gen in range(self.num_generations):
            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            best = population[0]
            avg_fitness = np.mean([ind.fitness for ind in population])
            diversity = self._compute_diversity(population)
            
            result.best_fitness_history.append(best.fitness)
            result.avg_fitness_history.append(avg_fitness)
            result.diversity_history.append(diversity)
            
            # Callback for progress reporting
            if self.callback:
                self.callback(gen, best, avg_fitness, diversity)
            
            # Step 3-6: Create next generation
            next_generation = []
            
            # Elitism: preserve top individuals
            for i in range(self.elitism_count):
                elite = Individual(weights=population[i].weights.copy())
                elite.fitness = population[i].fitness
                elite.sharpe = population[i].sharpe
                elite.annual_return = population[i].annual_return
                elite.annual_volatility = population[i].annual_volatility
                elite.max_drawdown = population[i].max_drawdown
                next_generation.append(elite)
            
            # Fill rest with offspring
            while len(next_generation) < self.population_size:
                # Selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                # Crossover
                child1, child2 = self._blx_crossover(parent1, parent2)
                
                # Mutation
                child1 = self._gaussian_mutate(child1)
                child2 = self._gaussian_mutate(child2)
                
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
            
            # Evaluate new population
            population = next_generation
            self._evaluate_population(population)
        
        # Final sort
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        result.best_individual = population[0]
        result.final_population = population[:20]  # Top 20
        result.generations_run = self.num_generations
        
        return result


def optimize_portfolio(returns_df, symbols, callback=None, **kwargs):
    """
    Convenience function to run GA portfolio optimization.
    
    Args:
        returns_df: DataFrame of daily returns
        symbols: List of asset symbols
        callback: Optional progress callback(gen, best, avg_fitness, diversity)
        **kwargs: Override GA parameters
        
    Returns:
        GAResult with optimal portfolio weights
    """
    returns_array = returns_df.values
    n_assets = len(symbols)
    
    ga = GeneticAlgorithm(
        returns=returns_array,
        n_assets=n_assets,
        callback=callback,
        **kwargs
    )
    
    result = ga.run()
    return result
