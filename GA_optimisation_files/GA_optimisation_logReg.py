import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import f1_score
from scipy.special import expit
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/1TB/wYr_model') 

from wYr_model_original import f_generate_TARGET_dataset

def evaluate_weights(X, y, weights):
    """Evaluate weights using logistic regression approach"""
    weights_array = np.array([float(w) for w in weights])
    logits = np.dot(X, weights)
    probs = expit(logits)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(y, preds)
    return f1

def fitness(weights, X, y, n_bootstrap=100, reg_strength=0.01):
    """Fitness function using bootstrap sampling"""
    f1s = []

    for i in range(n_bootstrap):
        if i % 50 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")

        X_sample, y_sample = resample(X, y, replace=True, n_samples=250, random_state=42*i)
        
        try:
            f1 = evaluate_weights(X_sample, y_sample, weights)
            if f1 is not None and not np.isnan(f1):
                f1s.append(f1)
        except Exception as e:
            continue
        
    if len(f1s) == 0:
        return 0.0
    
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    reg_penalty = np.sum(np.array(weights) ** 2)
    fitness_score = (mean_f1 / std_f1) - reg_strength * reg_penalty

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
    """Generate initial population with empirical weights as seed"""
    population = [np.array(empirical_weights).copy()]
    seed_percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for _ in range(size - 1):
        percents = np.random.choice(seed_percentages, size=len(empirical_weights))
        signs = np.random.choice([-1, 1], size=len(empirical_weights))
        changes = signs * (percents / 100.0) * np.array(empirical_weights)
        new_individual = np.array(empirical_weights) + changes
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

# def adaptive_mutation(individual, base_mutation_rate, generation, max_generations, diversity):
#     """Adaptive mutation that increases when diversity is low"""
#     mutated = individual.copy()
    
#     # Increase mutation rate when diversity is low
#     diversity_factor = max(1.0, 1.5 - diversity / 20.0)  # Adjust based on your diversity range
    
#     # Increase mutation rate in later generations
#     generation_factor = 1.0 + (generation / max_generations) * 0.3
    
#     effective_mutation_rate = base_mutation_rate * diversity_factor * generation_factor
#     effective_mutation_rate = min(effective_mutation_rate, 0.5)  # Cap at 50%
    
#     for i in range(len(individual)):
#         if np.random.rand() < effective_mutation_rate:
#             # Variable mutation strength
#             mutation_strength = np.random.choice([-5, -3, -2, -1, 1, 2, 3, 5], 
#                                                p=[0.1, 0.15, 0.2, 0.25, 0.1, 0.1, 0.05, 0.05])
#             mutated[i] = np.clip(individual[i] + mutation_strength, 0, 100)
    
#     return mutated

def adaptive_mutation(individual, base_mutation_rate, generation, max_generations, diversity):
    """Adaptive mutation that increases when diversity is low and uses percent-based mutation."""
    mutated = individual.copy()
    seed_percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Increase mutation rate when diversity is low
    diversity_factor = max(1.0, 1.5 - diversity / 20.0)
    # Increase mutation rate in later generations
    generation_factor = 1.0 + (generation / max_generations) * 0.3
    effective_mutation_rate = base_mutation_rate * diversity_factor * generation_factor
    effective_mutation_rate = min(effective_mutation_rate, 0.5)  # Cap at 50%
    
    for i in range(len(individual)):
        if np.random.rand() < effective_mutation_rate:
            percent = np.random.choice(seed_percentages)
            sign = np.random.choice([-1, 1])
            change = sign * (percent / 100.0) * mutated[i]
            mutated[i] = mutated[i] + change
    return mutated

# def inject_immigrants(population, empirical_weights, num_immigrants=10):
#     """Inject random individuals to maintain diversity"""
#     immigrants = []
#     for _ in range(num_immigrants):
#         immigrant = np.random.randint(-1, 2, size=len(empirical_weights))
#         immigrants.append(immigrant)
    
#     # Replace worst individuals with immigrants
#     return np.array(immigrants)

def genetic_algorithm_improved(initial_population, population_size, generations, base_mutation_rate, 
                             X, y, parameters, n_bootstrap=100, reg_strength=0.01):
    """
    Improved genetic algorithm with updated fitness function
    
    Args:
        initial_population: Initial weights (empirical weights)
        population_size: Size of the population
        generations: Number of generations
        base_mutation_rate: Base mutation rate
        X: Feature matrix
        y: Target labels
        parameters: List of parameter names
        penalty_lambda: Penalty for standard deviation in fitness
        n_bootstrap: Number of bootstrap samples
    """
    
    population = genPopulation(population_size, initial_population)
    best_individual = None
    best_fitness = float('-inf')
    
    # Track convergence
    fitness_history = []
    diversity_history = []
    stagnation_counter = 0
    fitness_cache = {}

    for generation in range(generations):
        current_fitness_scores = []
        
        for i, ind in enumerate(population):
            key = tuple(ind)
            if key not in fitness_cache:
                print(f"Evaluating fitness for individual {i+1}/{len(population)} in generation {generation+1}")
                fitness_cache[key] = fitness(ind, X, y, n_bootstrap=n_bootstrap, reg_strength=0.01)
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
        
        # # Add immigrants if diversity is low or stagnation is detected
        # if diversity < 3.0 or stagnation_counter > 50:
        #     immigrants = inject_immigrants(population, initial_population, num_immigrants=5)
        #     next_generation.extend(immigrants)
        #     print(f"  -> Injected immigrants due to {'low diversity' if diversity < 3.0 else 'stagnation'}")
        #     # Reset stagnation counter partially when injecting immigrants
        #     if stagnation_counter > 30:
        #         stagnation_counter = max(0, stagnation_counter - 10)
        
        # Fill remaining slots randomly
        seed_percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        while len(next_generation) < population_size:
            percents = np.random.choice(seed_percentages, size=len(empirical_weights))
            signs = np.random.choice([-1, 1], size=len(empirical_weights))
            changes = signs * (percents / 100.0) * np.array(empirical_weights)
            random_individual = np.array(empirical_weights) + changes
            next_generation.append(random_individual)
        
        population = np.array(next_generation[:population_size])
        
        # # Population restart if completely stagnated
        # if stagnation_counter > 50:
        #     print("  -> Population restart due to prolonged stagnation")
        #     # Keep only the best few individuals (ensure we don't exceed available elites)
        #     num_keep = min(10, len(elite_indices))
        #     population[:num_keep] = population[elite_indices[:num_keep]]
        #     # Regenerate the rest
        #     for i in range(num_keep, population_size):
        #         new_individual = initial_population + np.random.randint(-12, 13, size=len(initial_population))
        #         population[i] = np.clip(new_individual, 0, 100)
        #     stagnation_counter = 0
        
    return best_individual, best_fitness, fitness_history, diversity_history

# Usage with updated function signature
if __name__ == "__main__":
    paths1 = {}
    paths1['ZM'] = "/1TB/wYr_model/wYr_datasets/training_ZM.xlsx"
    paths1['Questionnaires'] = "/1TB/wYr_model/wYr_datasets/training_questionnaire.xlsx"
    paths1['Prospective'] = "/1TB/wYr_model/wYr_datasets/training_followup.xlsx"
    paths1["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    paths2 = {}
    paths2['ZM'] = "/1TB/wYr_model/wYr_datasets/tuneing_ZM.xlsx"
    paths2['Questionnaires'] = "/1TB/wYr_model/wYr_datasets/tuneing_questionnaire.xlsx"
    paths2['Prospective'] = "/1TB/wYr_model/wYr_datasets/tuneing_followup.xlsx"

    target1 = f_generate_TARGET_dataset(paths1, 4)
    target2 = f_generate_TARGET_dataset(paths2, 4)

    df_base_train = target1.dataset.copy()
    df_base_tune = target2.dataset.copy()

    df_full = pd.concat([df_base_train, df_base_tune], ignore_index=True)
    df_meta = pd.read_excel(paths1['Model_information'], sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()
    feature_cols = [
        "Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", 
        "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", 
        "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed", 
        "ICONFES_Score_Adjusted", "IPAQ_Cat", "Age", "Fall_2"
    ]

    X = df_full[feature_cols].values
    y = df_full["label"].values

    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df["label"] = y

    df_majority = train_df[train_df["label"] == 0]  # no fall
    df_minority = train_df[train_df["label"] == 1]  # fall

    df_majority_downsampled = resample(df_majority, 
                                       replace=False, 
                                       n_samples=len(df_minority), 
                                       random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train = df_balanced[feature_cols].values
    y_train = df_balanced["label"].values

    print(f"Final training set shape: {X_train.shape}, Positive class count: {np.sum(y_train)}, Negative class count: {len(y_train) - np.sum(y_train)}")

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)

    # Empirical weights to initialize GA
    empirical_weights = [ 0.31827473, 0.24375703,-0.13164474,-0.00125984, 0.59986035,-0.5462613,
  -0.33924535,-0.07833128,-0.20310523, 0.60835199, 0.15427231, 0.09682677,
   0.06399634,-0.0267412,  0.18261466, 0.31665181]

    results = []

    best_weights, best_fitness, fitness_history, diversity_history = genetic_algorithm_improved(
        initial_population=empirical_weights,
        population_size=200,
        generations=200,
        base_mutation_rate=0.15,
        X=X_scaled_train,
        y=y_train,
        parameters=parameters,
        n_bootstrap=200,
        reg_strength=0.01
    )

    # Final evaluation (on training data, since no validation set)
    train_f1 = evaluate_weights(X_scaled_train, y_train, best_weights)
    print(f"\n✅ Final F1 Score on Training Data: {train_f1:.4f}")
    results.append((best_weights, train_f1))

    best_result = max(results, key=lambda x: x[1])
    best_weights, best_f1_score = best_result

    print("\n✅ Best Weights from GA:")
    print(best_weights)
