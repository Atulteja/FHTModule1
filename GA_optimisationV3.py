import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

from wYr_model import f_riskscore, f_thresholding_predict, f_evaluate_predictions, f_generate_TARGET_dataset

def fitness(weights, df_base, parameters, mean_vals, std_vals, dirns, threshold=64, n_std=4):
    df = df_base.copy()
    weights = {param: int(w) for param, w in zip(parameters, weights)}

    df["risk score"] = 0
    for var in parameters:
        mean = mean_vals[var]
        std = std_vals[var]

        if dirns[var] == "<":
            th_low = mean - (n_std * std)
            df["point"] = df[var] < th_low
        elif dirns[var] == ">":
            th_high = mean + (n_std * std)
            df["point"] = df[var] > th_high
        elif dirns[var] == "=":
            th_high = mean + (n_std * std)
            df["point"] = df[var] == th_high
        elif dirns[var] == "><":
            th_low = max(0, mean - (n_std * std)) if "Var" in var else mean - (n_std * std)
            th_high = mean + (n_std * std)
            df["point"] = ~df[var].between(th_low, th_high)
        else:
            df["point"] = 0

        df["point"] = df["point"].astype(int) * weights[var]
        df["risk score"] += df["point"]

    df.drop(columns=["point"], inplace=True)
    df["prediction"] = (df["risk score"] >= threshold).astype(int)

    return 1 - roc_auc_score(df["label"].values, df["prediction"].values)

def calculate_diversity(population):
    """Calculate population diversity as average pairwise distance"""
    if len(population) < 2:
        return 0
    
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Use Manhattan distance instead of Euclidean for better diversity measurement
            distance = np.sum(np.abs(population[i] - population[j]))
            distances.append(distance)
    
    return np.mean(distances)

def calculate_fitness_diversity(fitness_scores):
    """Calculate diversity in fitness values"""
    return np.std(fitness_scores)

def genPopulation(size, empirical_weights):
    """Generate initial population with better diversity"""
    population = [empirical_weights.copy()]
    
    # Create more diverse initial population
    for _ in range(size - 1):
        # Use different strategies for generating diverse individuals
        strategy = np.random.choice(['gaussian', 'uniform', 'focused'])
        
        if strategy == 'gaussian':
            noise = np.random.normal(0, 10, size=len(empirical_weights))
            new_individual = empirical_weights + noise.astype(int)
        elif strategy == 'uniform':
            new_individual = np.random.uniform(0, 50, size=len(empirical_weights)).astype(int)
        else:  # focused - randomly boost some parameters
            new_individual = empirical_weights.copy()
            boost_indices = np.random.choice(len(empirical_weights), 
                                           size=np.random.randint(1, len(empirical_weights)//2), 
                                           replace=False)
            for idx in boost_indices:
                new_individual[idx] += np.random.randint(-15, 16)
        
        new_individual = np.clip(new_individual, 0, 100)
        population.append(new_individual)
    
    return np.array(population)

def novelty_selection(population, fitness_scores, num_selected, novelty_weight=0.3):
    """Select individuals based on both fitness and novelty"""
    selected_indices = []
    
    # Calculate novelty scores for each individual
    novelty_scores = np.zeros(len(population))
    for i in range(len(population)):
        distances = []
        for j in range(len(population)):
            if i != j:
                distance = np.sum(np.abs(population[i] - population[j]))
                distances.append(distance)
        novelty_scores[i] = np.mean(distances) if distances else 0
    
    # Normalize scores
    if np.max(novelty_scores) > 0:
        novelty_scores = novelty_scores / np.max(novelty_scores)
    
    fitness_scores_norm = fitness_scores / np.max(fitness_scores) if np.max(fitness_scores) > 0 else fitness_scores
    
    # Combined score (lower is better for fitness, higher is better for novelty)
    combined_scores = (1 - novelty_weight) * fitness_scores_norm - novelty_weight * novelty_scores
    
    # Select based on combined score
    selected_indices = np.argsort(combined_scores)[:num_selected]
    return population[selected_indices]

def rank_based_selection(population, fitness_scores, num_selected, selection_pressure=1.5):
    """Rank-based selection to maintain diversity"""
    # Rank individuals (best rank = 0)
    ranks = np.argsort(np.argsort(fitness_scores))
    
    # Calculate selection probabilities based on rank
    probabilities = []
    for rank in ranks:
        prob = (2 - selection_pressure) + 2 * (selection_pressure - 1) * (len(ranks) - rank - 1) / (len(ranks) - 1)
        probabilities.append(prob / len(ranks))
    
    probabilities = np.array(probabilities)
    probabilities = probabilities / np.sum(probabilities)  # Normalize
    
    # Select individuals based on probabilities
    selected_indices = np.random.choice(len(population), size=num_selected, 
                                      replace=True, p=probabilities)
    return population[selected_indices]

def multi_point_crossover(parent1, parent2, num_points=None):
    """Multi-point crossover with variable number of crossover points"""
    if num_points is None:
        num_points = np.random.randint(2, min(6, len(parent1)))
    
    crossover_points = sorted(np.random.choice(range(1, len(parent1)), 
                                             min(num_points, len(parent1)-1), 
                                             replace=False))
    
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Alternate segments
    for i, point in enumerate(crossover_points):
        if i == 0:
            start = 0
        else:
            start = crossover_points[i-1]
        
        if i % 2 == 1:  # Odd segments - swap
            child1[start:point] = parent2[start:point]
            child2[start:point] = parent1[start:point]
    
    # Handle last segment
    if len(crossover_points) % 2 == 1:
        child1[crossover_points[-1]:] = parent2[crossover_points[-1]:]
        child2[crossover_points[-1]:] = parent1[crossover_points[-1]:]
    
    return child1, child2

def adaptive_mutation_v2(individual, generation, max_generations, diversity, fitness_diversity):
    """Improved adaptive mutation"""
    mutated = individual.copy()
    
    # Base mutation rate that increases over time
    base_rate = 0.1 + 0.3 * (generation / max_generations)
    
    # Diversity factor - increase mutation when diversity is low
    diversity_factor = max(1.0, 3.0 - diversity / 20.0)
    
    # Fitness diversity factor - increase mutation when fitness diversity is low
    fitness_diversity_factor = max(1.0, 2.0 - fitness_diversity * 10)
    
    # Calculate effective mutation rate
    mutation_rate = base_rate * diversity_factor * fitness_diversity_factor
    mutation_rate = min(mutation_rate, 0.6)  # Cap at 60%
    
    # Apply mutation
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # Variable mutation strength based on generation
            if generation < max_generations * 0.3:  # Early generations - larger mutations
                mutation_strength = np.random.choice([-10, -7, -5, -3, -1, 1, 3, 5, 7, 10])
            elif generation < max_generations * 0.7:  # Mid generations - medium mutations
                mutation_strength = np.random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
            else:  # Late generations - fine-tuning
                mutation_strength = np.random.choice([-3, -2, -1, 1, 2, 3])
            
            mutated[i] = np.clip(individual[i] + mutation_strength, 0, 100)
    
    return mutated

def create_diverse_immigrants(empirical_weights, num_immigrants, generation, max_generations):
    """Create diverse immigrants with different strategies"""
    immigrants = []
    
    for i in range(num_immigrants):
        if i % 3 == 0:  # Random exploration
            immigrant = np.random.randint(0, 101, size=len(empirical_weights))
        elif i % 3 == 1:  # Based on empirical but with large variation
            noise_scale = 15 if generation < max_generations * 0.5 else 25
            immigrant = empirical_weights + np.random.randint(-noise_scale, noise_scale+1, 
                                                            size=len(empirical_weights))
            immigrant = np.clip(immigrant, 0, 100)
        else:  # Focused boost strategy
            immigrant = empirical_weights.copy()
            # Randomly select parameters to boost significantly
            boost_count = np.random.randint(1, len(empirical_weights)//3 + 1)
            boost_indices = np.random.choice(len(empirical_weights), boost_count, replace=False)
            for idx in boost_indices:
                immigrant[idx] = np.random.randint(max(0, immigrant[idx]-10), 101)
        
        immigrants.append(immigrant)
    
    return np.array(immigrants)

def genetic_algorithm_improved_v2(initial_population, population_size, generations, 
                                df_base, parameters, mean_vals, std_vals, dirns, threshold=64):
    
    population = genPopulation(population_size, initial_population)
    best_individual = None
    best_fitness = float('inf')
    
    # Track convergence and diversity
    fitness_history = []
    diversity_history = []
    fitness_diversity_history = []
    stagnation_counter = 0
    best_fitness_window = []
    
    for generation in range(generations):
        # Calculate fitness for all individuals
        fitness_scores = np.array([
            fitness(ind, df_base, parameters, mean_vals, std_vals, dirns, threshold)
            for ind in population
        ])
        
        # Calculate diversity metrics
        diversity = calculate_diversity(population)
        fitness_diversity = calculate_fitness_diversity(fitness_scores)
        
        diversity_history.append(diversity)
        fitness_diversity_history.append(fitness_diversity)
        
        # Track best individual and stagnation
        best_idx = np.argmin(fitness_scores)
        current_best = fitness_scores[best_idx]
        
        if current_best < best_fitness:
            best_fitness = current_best
            best_individual = population[best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Track fitness improvement over a window
        best_fitness_window.append(current_best)
        if len(best_fitness_window) > 20:
            best_fitness_window.pop(0)
        
        fitness_history.append(best_fitness)
        
        print(f"Generation {generation + 1}: Best={best_fitness:.6f}, Current={current_best:.6f}, "
              f"Diversity={diversity:.2f}, FitDiv={fitness_diversity:.4f}, "
              f"Stagnation={stagnation_counter}")
        
        # Selection strategy based on diversity and stagnation
        selection_size = max(population_size // 2, 50)
        
        if diversity < 10.0 or fitness_diversity < 0.001:  # Low diversity
            selected = novelty_selection(population, fitness_scores, selection_size, 
                                       novelty_weight=0.4)
            print(f"  -> Using novelty selection (diversity={diversity:.2f})")
        elif stagnation_counter > 20:
            selected = rank_based_selection(population, fitness_scores, selection_size, 
                                          selection_pressure=1.2)
            print(f"  -> Using rank-based selection (stagnation={stagnation_counter})")
        else:
            # Standard tournament selection
            selected = []
            for _ in range(selection_size):
                tournament_size = np.random.randint(2, 5)
                participants = np.random.choice(len(population), tournament_size, replace=False)
                best_idx = participants[np.argmin(fitness_scores[participants])]
                selected.append(population[best_idx])
            selected = np.array(selected)
        
        # Create next generation
        next_generation = []
        
        # Elitism - but vary elite size based on diversity
        if diversity > 15:
            elite_size = max(3, population_size // 15)
        else:
            elite_size = max(1, population_size // 30)  # Fewer elites when diversity is low
        
        elite_indices = np.argsort(fitness_scores)[:elite_size]
        for idx in elite_indices:
            next_generation.append(population[idx].copy())
        
        # Generate offspring
        immigrant_slots = 0
        if diversity < 8.0 or stagnation_counter > 25:
            immigrant_slots = max(5, population_size // 20)
        
        target_offspring = population_size - len(next_generation) - immigrant_slots
        
        while len(next_generation) < len(next_generation) + target_offspring:
            # Select parents
            parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
            
            # Crossover
            if np.random.rand() < 0.8:  # Crossover probability
                child1, child2 = multi_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = adaptive_mutation_v2(child1, generation, generations, 
                                        diversity, fitness_diversity)
            child2 = adaptive_mutation_v2(child2, generation, generations, 
                                        diversity, fitness_diversity)
            
            next_generation.append(child1)
            if len(next_generation) < len(next_generation) + target_offspring:
                next_generation.append(child2)
        
        # Add immigrants if needed
        if immigrant_slots > 0:
            immigrants = create_diverse_immigrants(initial_population, immigrant_slots, 
                                                 generation, generations)
            next_generation.extend(immigrants)
            print(f"  -> Added {immigrant_slots} diverse immigrants")
        
        # Fill remaining slots if needed
        while len(next_generation) < population_size:
            random_individual = create_diverse_immigrants(initial_population, 1, 
                                                        generation, generations)[0]
            next_generation.append(random_individual)
        
        population = np.array(next_generation[:population_size])
        
        # Major restart if severely stagnated
        if stagnation_counter > 80:
            print("  -> MAJOR RESTART: Severe stagnation detected")
            # Keep only the very best
            num_keep = min(5, population_size // 20)
            best_individuals = population[np.argsort(fitness_scores)[:num_keep]]
            
            # Regenerate population
            population = genPopulation(population_size, initial_population)
            population[:num_keep] = best_individuals
            stagnation_counter = 0
        
        # Check for convergence based on fitness window
        if len(best_fitness_window) == 20:
            fitness_improvement = best_fitness_window[0] - best_fitness_window[-1]
            if fitness_improvement < 1e-6 and generation > 100:
                print(f"  -> Potential convergence detected (improvement: {fitness_improvement:.8f})")
    
    return best_individual, best_fitness, fitness_history, diversity_history

# Usage
if __name__ == "__main__":
    paths = {}
    paths['ZM'] = "//1TB/Dataset_Feb2025/TARGETZMParameters_All_20241015.xlsx"
    paths['Questionnaires'] = "/1TB/Dataset_Feb2025/TARGET 5 November 2024 dataset to SEC_051124 numeric n=2291.xlsx"
    paths['Prospective'] = "/1TB/Dataset_Feb2025/TARGET follow-up 18.02.2025 to SEC 26022025 numeric.xls"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    # Load once
    target = f_generate_TARGET_dataset(paths, 4)
    df_base = target.dataset.copy()

    df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()
    mean_vals = {k: float(g["Mean/cutoff"].iloc[0]) for k, g in df_meta.groupby("Parameter")}
    std_vals = {k: float(g["StdDev"].iloc[0]) for k, g in df_meta.groupby("Parameter")}
    dirns = {k: str(g["Faller_if"].iloc[0]) for k, g in df_meta.groupby("Parameter")}
    empirical_weights = df_meta["Weights"].values.tolist()

    best_weights, best_fitness, fitness_history, diversity_history = genetic_algorithm_improved_v2(
        initial_population=empirical_weights,
        population_size=200,  # Larger population
        generations=500,     # More generations
        df_base=df_base,
        parameters=parameters,
        mean_vals=mean_vals,
        std_vals=std_vals,
        dirns=dirns,
        threshold=64
    )

    print("Best Weights:", best_weights)
    print("Best Fitness:", best_fitness)
    
    # Plot convergence and diversity
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(fitness_history)
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    
    plt.subplot(1, 2, 2)
    plt.plot(diversity_history)
    plt.title('Diversity Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Population Diversity')
    
    plt.tight_layout()
    plt.show()