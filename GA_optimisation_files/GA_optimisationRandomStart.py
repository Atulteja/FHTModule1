import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.utils import resample

sys.path.insert(0, '/1TB/wYr_model') 

from wYr_model_original import f_generate_TARGET_dataset

# def fitness(weights, df_base, parameters, mean_vals, std_vals, dirns, threshold=64, n_std=4):
#     df = df_base.copy()
#     weights = {param: int(w) for param, w in zip(parameters, weights)}

#     df["risk score"] = 0
#     for var in parameters:
#         mean = mean_vals[var]
#         std = std_vals[var]

#         if dirns[var] == "<":
#             th_low = mean - (n_std * std)
#             df["point"] = df[var] < th_low
#         elif dirns[var] == ">":
#             th_high = mean + (n_std * std)
#             df["point"] = df[var] > th_high
#         elif dirns[var] == "=":
#             th_high = mean + (n_std * std)
#             df["point"] = df[var] == th_high
#         elif dirns[var] == "><":
#             th_low = max(0, mean - (n_std * std)) if "Var" in var else mean - (n_std * std)
#             th_high = mean + (n_std * std)
#             df["point"] = ~df[var].between(th_low, th_high)
#         else:
#             df["point"] = 0

#         df["point"] = df["point"].astype(int) * weights[var]
#         df["risk score"] += df["point"]

#     df.drop(columns=["point"], inplace=True)
#     df["prediction"] = (df["risk score"] >= threshold).astype(int)

#     return 1 - roc_auc_score(df["label"].values, df["prediction"].values)

def evaluate_weights(df, weights, parameters, mean_vals, std_vals, dirns, threshold=64, n_std=4):
    """Evaluate the fitness of a set of weights by calculating the risk score and AUC"""
    df = df.copy()
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

    return f1_score(df["label"].values, df["prediction"].values)

def fitness(weights, df_base, parameters, mean_vals, std_vals, dirns, threshold=64, n_std=4, penalty_lambda=0.1, n_bootstrap=100):
    weights = {param: int(w) for param, w in zip(parameters, weights)}
    weights_vec = np.array([weights[param] for param in parameters])
    f1s = []

    for i in range(n_bootstrap):
        if i % 50 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")

        df_sample = resample(df_base, replace=True, n_samples=100, random_state=42*i)
        
        try:
            f1 = evaluate_weights(df_sample, weights, parameters, mean_vals, std_vals, dirns, threshold, n_std)
            if f1 is not None and not np.isnan(f1):
                f1s.append(f1)
        except Exception as e:
            continue
        
    if len(f1s) == 0:
        return 0.0
    
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    fitness_score = mean_f1/ std_f1

    return max(0, fitness_score)


def calculate_diversity(population):
    """Calculate population diversity as average pairwise distance"""
    if len(population) < 2:
        return 0
    
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = np.linalg.norm(population[i] - population[j])
            distances.append(distance)
    
    return np.mean(distances)

def genPopulation(size, empirical_weights):
    population = [empirical_weights.copy()]
    for _ in range(size - 1):
        new_individual = np.random.randint(0, 101, size=len(empirical_weights))
        population.append(new_individual)
    return np.array(population)

def tournament_selection(population, fitness_scores, num_selected, tournament_size=3):
    """Tournament selection with smaller tournament size to maintain diversity"""
    selected = []
    for _ in range(num_selected):
        participants = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = participants[np.argmax(fitness_scores[participants])]
        selected.append(population[best_idx])
    return np.array(selected)

def diversity_selection(population, fitness_scores, num_selected, diversity_weight=0.3):
    """Combine fitness and diversity for selection"""
    selected_indices = []
    
    # First, select some based purely on fitness
    fitness_based = int(num_selected * (1 - diversity_weight))
    best_indices = np.argsort(-fitness_scores)[:fitness_based]
    selected_indices.extend(best_indices)
    
    # Then select remaining based on diversity
    remaining_indices = list(set(range(len(population))) - set(selected_indices))
    
    for _ in range(num_selected - len(selected_indices)):
        if not remaining_indices:
            break
            
        # Calculate diversity contribution of each remaining individual
        diversity_scores = []
        for idx in remaining_indices:
            # Calculate minimum distance to already selected individuals
            min_distance = float('inf')
            for sel_idx in selected_indices:
                distance = np.linalg.norm(population[idx] - population[sel_idx])
                min_distance = min(min_distance, distance)
            diversity_scores.append(min_distance)
        
        # Select the one that adds most diversity
        best_diversity_idx = remaining_indices[np.argmax(diversity_scores)]
        selected_indices.append(best_diversity_idx)
        remaining_indices.remove(best_diversity_idx)
    
    return population[selected_indices]

def adaptive_crossover(parent1, parent2, generation, max_generations):
    """Adaptive crossover that becomes more explorative over time"""
    # Multiple crossover points for better mixing
    num_crossover_points = min(3, len(parent1) // 4)
    crossover_points = sorted(np.random.choice(range(1, len(parent1)), num_crossover_points, replace=False))
    
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    for i, point in enumerate(crossover_points):
        if i % 2 == 0:  # Alternate which parent contributes
            child1[point:] = parent2[point:]
            child2[point:] = parent1[point:]
    
    return child1, child2

def adaptive_mutation(individual, base_mutation_rate, generation, max_generations, diversity):
    """Adaptive mutation that increases when diversity is low"""
    mutated = individual.copy()
    
    # Increase mutation rate when diversity is low
    diversity_factor = max(1.0, 1.5 - diversity / 20.0)  # Adjust based on your diversity range
    
    # Increase mutation rate in later generations
    generation_factor = 1.0 + (generation / max_generations) * 0.3
    
    effective_mutation_rate = base_mutation_rate * diversity_factor * generation_factor
    effective_mutation_rate = min(effective_mutation_rate, 0.5)  # Cap at 50%
    
    for i in range(len(individual)):
        if np.random.rand() < effective_mutation_rate:
            # Variable mutation strength
            mutation_strength = np.random.choice([-5, -3, -2, -1, 1, 2, 3, 5], 
                                               p=[0.1, 0.15, 0.2, 0.25, 0.1, 0.1, 0.05, 0.05])
            mutated[i] = np.clip(individual[i] + mutation_strength, 0, 100)
    
    return mutated

def inject_immigrants(population, empirical_weights, num_immigrants=10):
    """Inject random individuals to maintain diversity"""
    immigrants = []
    for _ in range(num_immigrants):
        immigrant = np.random.randint(0, 101, size=len(empirical_weights))
        immigrant = np.clip(immigrant, 0, 100)
        immigrants.append(immigrant)
    
    # Replace worst individuals with immigrants
    return np.array(immigrants)

def genetic_algorithm_improved(initial_population, population_size, generations, base_mutation_rate, 
                             df_base, parameters, mean_vals, std_vals, dirns, threshold=64, penalty_lambda=1.0, n_bootstrap=100):
    
    population = genPopulation(population_size, initial_population)
    best_individual = None
    best_fitness = float('-inf')
    
    # Track convergence
    fitness_history = []
    diversity_history = []
    stagnation_counter = 0
    fitness_cache = {}
    fitness_scores = []   

    for generation in range(generations):
        current_fitness_scores = []  # Use a different variable name each generation
        
        for i, ind in enumerate(population):
            key = tuple(ind)
            if key not in fitness_cache:
                print(f"Evaluating fitness for individual {i+1}/{len(population)} in generation {generation+1}")
                fitness_cache[key] = fitness(ind, df_base, parameters, mean_vals, std_vals, dirns, threshold,
                                    n_std=4, penalty_lambda=penalty_lambda, n_bootstrap=n_bootstrap)
            current_fitness_scores.append(fitness_cache[key])
        
        fitness_scores = np.array(current_fitness_scores)
        
        # Calculate diversity
        diversity = calculate_diversity(population)
        diversity_history.append(diversity)
        
        # Track best individual
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_individual = population[best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        fitness_history.append(best_fitness)
        
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.6f}, "
                f"Diversity: {diversity:.2f}, Avg Fitness: {np.mean(fitness_scores):.6f}")
        
        # Selection - use larger selection pool and diversity-aware selection
        selection_size = max(population_size // 2, 40)
        
        if diversity < 5.0:  # Low diversity threshold - adjust based on your problem
            selected = diversity_selection(population, fitness_scores, selection_size, diversity_weight=0.4)
        else:
            selected = tournament_selection(population, fitness_scores, selection_size, tournament_size=3)
        
        # Create next generation
        next_generation = []
        
        # Elitism - keep best individuals
        elite_size = max(2, population_size // 20)
        elite_indices = np.argsort(-fitness_scores)[:elite_size]
        for idx in elite_indices:
            next_generation.append(population[idx].copy())
        
        # Generate offspring
        while len(next_generation) < population_size - 5:  # Leave room for immigrants
            parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
            child1, child2 = adaptive_crossover(parent1, parent2, generation, generations)
            
            child1 = adaptive_mutation(child1, base_mutation_rate, generation, generations, diversity)
            child2 = adaptive_mutation(child2, base_mutation_rate, generation, generations, diversity)
            
            next_generation.append(child1)
            if len(next_generation) < population_size - 5:
                next_generation.append(child2)
        
        # Add immigrants if diversity is low or stagnation is detected
        if diversity < 3.0 or stagnation_counter > 30:
            immigrants = inject_immigrants(population, initial_population, num_immigrants=5)
            next_generation.extend(immigrants)
            print(f"  -> Injected immigrants due to {'low diversity' if diversity < 3.0 else 'stagnation'}")
            # Reset stagnation counter partially when injecting immigrants
            if stagnation_counter > 30:
                stagnation_counter = max(0, stagnation_counter - 10)
        
        # Fill remaining slots randomly
        while len(next_generation) < population_size:
            random_individual = initial_population + np.random.randint(-5, 6, size=len(initial_population))
            random_individual = np.clip(random_individual, 0, 100)
            next_generation.append(random_individual)
        
        population = np.array(next_generation[:population_size])
        
        # Population restart if completely stagnated
        if stagnation_counter > 50:
            print("  -> Population restart due to prolonged stagnation")
            # Keep only the best few individuals (ensure we don't exceed available elites)
            num_keep = min(10, len(elite_indices))
            population[:num_keep] = population[elite_indices[:num_keep]]
            # Regenerate the rest
            for i in range(num_keep, population_size):
                new_individual = initial_population + np.random.randint(-12, 13, size=len(initial_population))
                population[i] = np.clip(new_individual, 0, 100)
            stagnation_counter = 0
        
    return best_individual, best_fitness, fitness_history, diversity_history

# Usage remains the same but with the improved function
if __name__ == "__main__":
    paths = {}
    paths['ZM'] = "/1TB/wYr_model/wYr_datasets/training_ZM.xlsx"
    paths['Questionnaires'] = "/1TB/wYr_model/wYr_datasets/training_questionnaire.xlsx"
    paths['Prospective'] = "/1TB/wYr_model/wYr_datasets/training_followup.xlsx"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    # Load once
    target = f_generate_TARGET_dataset(paths, 4)
    df_base = target.dataset.copy()
    df_majority = df_base[df_base["label"] == 0]
    df_minority = df_base[df_base["label"] == 1]
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()
    mean_vals = {k: float(g["Mean/cutoff"].iloc[0]) for k, g in df_meta.groupby("Parameter")}
    std_vals = {k: float(g["StdDev"].iloc[0]) for k, g in df_meta.groupby("Parameter")}
    dirns = {k: str(g["Faller_if"].iloc[0]) for k, g in df_meta.groupby("Parameter")}
    empirical_weights = df_meta["Weights"].values.tolist()

    results = []


    best_weights, best_fitness, fitness_history, diversity_history = genetic_algorithm_improved(
        initial_population=empirical_weights,
        population_size=200,
        generations=200,
        base_mutation_rate=0.15,
        df_base=df_balanced,
        parameters=parameters,
        mean_vals=mean_vals,
        std_vals=std_vals,
        dirns=dirns,
        threshold=64,
        penalty_lambda=0.1,
        n_bootstrap=50,
    )

    results.append(( best_fitness, best_weights))
    print(f"Best Fitness: {best_fitness:.4f}")

    best_result = max(results, key=lambda x: x[1])
    best_fitness, best_weights = best_result

    print("\n=============================")
    print(f"Best Fitness: {best_fitness:.4f}")
    print("Best Weights:", best_weights)