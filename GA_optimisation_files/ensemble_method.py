import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import f1_score
from scipy.special import expit
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.model_selection import KFold

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

def cross_dataset_fitness(weights, X_train, y_train, X_val, y_val, penalty_lambda=0.1, n_bootstrap=50):
    """Fitness function that evaluates across both training and validation datasets"""
    train_f1s = []
    val_f1s = []

    # Bootstrap evaluation on training set
    for i in range(n_bootstrap):
        X_sample, y_sample = resample(X_train, y_train, replace=True, n_samples=min(200, len(X_train)), random_state=42*i)
        try:
            f1 = evaluate_weights(X_sample, y_sample, weights)
            if f1 is not None and not np.isnan(f1):
                train_f1s.append(f1)
        except:
            continue
    
    # Bootstrap evaluation on validation set
    for i in range(n_bootstrap):
        X_sample, y_sample = resample(X_val, y_val, replace=True, n_samples=min(200, len(X_val)), random_state=42*i + 1000)
        try:
            f1 = evaluate_weights(X_sample, y_sample, weights)
            if f1 is not None and not np.isnan(f1):
                val_f1s.append(f1)
        except:
            continue
    
    if len(train_f1s) == 0 or len(val_f1s) == 0:
        return 0.0
    
    # Calculate means and stds
    train_mean = np.mean(train_f1s)
    train_std = np.std(train_f1s)
    val_mean = np.mean(val_f1s)
    val_std = np.std(val_f1s)
    
    # Fitness emphasizes: 1) good performance on both datasets, 2) low variance, 3) small gap between train/val
    generalization_gap = abs(train_mean - val_mean)
    combined_mean = (train_mean + val_mean) / 2
    combined_std = (train_std + val_std) / 2
    
    fitness_score = combined_mean - penalty_lambda * combined_std - 0.2 * generalization_gap
    
    return max(0, fitness_score)

def kfold_fitness(weights, X, y, penalty_lambda=0.1, k=5, n_bootstrap=30):
    """Fitness function using k-fold cross validation for robustness"""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_f1s = []
    
    for train_idx, val_idx in kfold.split(X):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Bootstrap within each fold
        bootstrap_f1s = []
        for i in range(n_bootstrap):
            X_sample, y_sample = resample(X_fold_train, y_fold_train, replace=True, 
                                        n_samples=min(150, len(X_fold_train)), random_state=42*i)
            try:
                f1 = evaluate_weights(X_sample, y_sample, weights)
                if f1 is not None and not np.isnan(f1):
                    bootstrap_f1s.append(f1)
            except:
                continue
        
        if bootstrap_f1s:
            fold_f1s.extend(bootstrap_f1s)
    
    if len(fold_f1s) == 0:
        return 0.0
    
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    fitness_score = mean_f1 - penalty_lambda * std_f1
    
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

def find_consensus_weights(all_good_individuals, similarity_threshold=0.1):
    """Find consensus weights from multiple good individuals across different runs"""
    if len(all_good_individuals) == 0:
        return None
    
    # Cluster similar individuals
    clusters = []
    used = [False] * len(all_good_individuals)
    
    for i, individual in enumerate(all_good_individuals):
        if used[i]:
            continue
            
        cluster = [individual]
        used[i] = True
        
        for j, other in enumerate(all_good_individuals):
            if used[j]:
                continue
            
            # Calculate normalized distance
            distance = np.linalg.norm(individual - other) / np.sqrt(len(individual))
            if distance < similarity_threshold:
                cluster.append(other)
                used[j] = True
        
        clusters.append(cluster)
    
    # Find the largest cluster (most consensus)
    largest_cluster = max(clusters, key=len)
    
    # Return the centroid of the largest cluster
    consensus_weights = np.mean(largest_cluster, axis=0)
    cluster_size = len(largest_cluster)
    
    return consensus_weights, cluster_size, len(clusters)

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

def robust_genetic_algorithm(initial_population, population_size, generations, base_mutation_rate, 
                        X_train, y_train, X_val, y_val, parameters, penalty_lambda=0.1, 
                        fitness_mode='cross_dataset', n_runs=5):
    """
    Robust genetic algorithm that finds generalizable weights
    
    Args:
        fitness_mode: 'cross_dataset', 'kfold', or 'bootstrap'
        n_runs: Number of independent runs to find consensus
    """
    
    all_good_individuals = []
    all_run_results = []
    
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"Starting Run {run + 1}/{n_runs}")
        print(f"{'='*50}")
        
        population = genPopulation(population_size, initial_population)
        best_individual = None
        best_fitness = float('-inf')
        
        # Track top individuals from this run
        run_top_individuals = []
        fitness_cache = {}

        for generation in range(generations):
            current_fitness_scores = []
            
            for i, ind in enumerate(population):
                key = tuple(ind)
                if key not in fitness_cache:
                    if generation % 20 == 0:  # Reduce output frequency
                        print(f"Run {run+1}, Gen {generation+1}: Evaluating individual {i+1}/{len(population)}")
                    
                    if fitness_mode == 'cross_dataset':
                        fitness_val = cross_dataset_fitness(ind, X_train, y_train, X_val, y_val, penalty_lambda)
                    elif fitness_mode == 'kfold':
                        # Combine train and validation for k-fold
                        X_combined = np.vstack([X_train, X_val])
                        y_combined = np.hstack([y_train, y_val])
                        fitness_val = kfold_fitness(ind, X_combined, y_combined, penalty_lambda)
                    else:  # bootstrap
                        fitness_val = bootstrap_fitness(ind, X_train, y_train, penalty_lambda)
                    
                    fitness_cache[key] = fitness_val
                current_fitness_scores.append(fitness_cache[key])
            
            fitness_scores = np.array(current_fitness_scores)
            
            # Calculate diversity
            diversity = calculate_diversity(population)
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_individual = population[best_idx].copy()
            
            # Store top 10% individuals from each generation (after warmup)
            if generation >= generations // 4:  # After 25% warmup
                top_indices = np.argsort(-fitness_scores)[:max(1, population_size // 10)]
                for idx in top_indices:
                    if fitness_scores[idx] > 0.1:  # Minimum fitness threshold
                        run_top_individuals.append((population[idx].copy(), fitness_scores[idx]))
            
            if generation % 20 == 0:
                print(f"Run {run+1}, Gen {generation + 1}: Best Fitness: {best_fitness:.6f}, "
                    f"Diversity: {diversity:.2f}, Avg Fitness: {np.mean(fitness_scores):.6f}")
            
            # Selection and evolution (same as before)
            selection_size = max(population_size // 2, 40)
            
            if diversity < 5.0:
                selected = diversity_selection(population, fitness_scores, selection_size, diversity_weight=0.4)
            else:
                selected = tournament_selection(population, fitness_scores, selection_size, tournament_size=3)
            
            # Create next generation
            next_generation = []
            
            # Elitism
            elite_size = max(2, population_size // 20)
            elite_indices = np.argsort(-fitness_scores)[:elite_size]
            for idx in elite_indices:
                next_generation.append(population[idx].copy())
            
            # Generate offspring
            while len(next_generation) < population_size - 5:
                parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
                child1, child2 = adaptive_crossover(parent1, parent2, generation, generations)
                
                child1 = adaptive_mutation(child1, base_mutation_rate, generation, generations, diversity)
                child2 = adaptive_mutation(child2, base_mutation_rate, generation, generations, diversity)
                
                next_generation.append(child1)
                if len(next_generation) < population_size - 5:
                    next_generation.append(child2)
            
            # Fill remaining slots randomly
            seed_percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            while len(next_generation) < population_size:
                percents = np.random.choice(seed_percentages, size=len(initial_population))
                signs = np.random.choice([-1, 1], size=len(initial_population))
                changes = signs * (percents / 100.0) * np.array(initial_population)
                random_individual = np.array(initial_population) + changes
                next_generation.append(random_individual)
            
            population = np.array(next_generation[:population_size])
        
        # Store results from this run
        all_run_results.append({
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'top_individuals': run_top_individuals
        })
        
        # Add good individuals to overall pool
        sorted_top = sorted(run_top_individuals, key=lambda x: x[1], reverse=True)
        # Take top 20% from this run
        n_take = max(1, len(sorted_top) // 5)
        for ind, fit in sorted_top[:n_take]:
            all_good_individuals.append(ind)
        
        print(f"Run {run+1} completed. Best fitness: {best_fitness:.6f}")
        print(f"Added {n_take} individuals to consensus pool")
    
    # Find consensus weights
    print(f"\n{'='*50}")
    print("Finding Consensus Weights...")
    print(f"{'='*50}")
    
    consensus_weights, cluster_size, n_clusters = find_consensus_weights(all_good_individuals, similarity_threshold=0.15)
    
    print(f"Found {n_clusters} clusters from {len(all_good_individuals)} good individuals")
    print(f"Largest cluster has {cluster_size} individuals ({cluster_size/len(all_good_individuals)*100:.1f}%)")
    
    return consensus_weights, all_run_results, all_good_individuals

def bootstrap_fitness(weights, X, y, penalty_lambda=0.1, n_bootstrap=100):
    """Original bootstrap fitness function for comparison"""
    f1s = []

    for i in range(n_bootstrap):
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
    fitness_score = mean_f1 - penalty_lambda * std_f1

    return max(0, fitness_score)

# Usage
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

    target = f_generate_TARGET_dataset(paths1, 4)
    target2 = f_generate_TARGET_dataset(paths2, 4)
    df_base_validate = target2.dataset.copy()
    df_base_train = target.dataset.copy()
    df_meta = pd.read_excel(paths1['Model_information'], sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()

    feature_cols = ["Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", 
                              "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", 
                              "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed", "ICONFES_Score_Adjusted", "IPAQ_Cat", "Age", "Fall_2"]
    X_train = df_base_train[feature_cols].values
    y_train = df_base_train["label"].values
    X_validate = df_base_validate[feature_cols].values
    y_validate = df_base_validate["label"].values
    print(f"Feature matrix shape: {X_train.shape}, Labels shape: {y_train.shape}")

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_validate = scaler.transform(X_validate)  # Use transform, not fit_transform for validation
    
    empirical_weights = [ 0.31827473, 0.24375703,-0.13164474,-0.00125984, 0.59986035,-0.5462613,
  -0.33924535,-0.07833128,-0.20310523, 0.60835199, 0.15427231, 0.09682677,
   0.06399634,-0.0267412,  0.18261466, 0.31665181]
    
    # Test different fitness modes and penalty values
    fitness_modes = ['cross_dataset', 'kfold']
    penalty_values = [0.1, 0.3, 0.5]
    
    results = []
    
    for mode in fitness_modes:
        for penalty in penalty_values:
            print(f"\n{'#'*60}")
            print(f"Testing: {mode} with penalty_lambda={penalty}")
            print(f"{'#'*60}")
            
            consensus_weights, run_results, all_good = robust_genetic_algorithm(
                initial_population=empirical_weights,
                population_size=150,  # Smaller population for multiple runs
                generations=100,      # Fewer generations per run
                base_mutation_rate=0.15,
                X_train=X_scaled_train,
                y_train=y_train,
                X_val=X_scaled_validate,
                y_val=y_validate,
                parameters=parameters,
                penalty_lambda=penalty,
                fitness_mode=mode,
                n_runs=3  # Multiple runs for consensus
            )
            
            # Evaluate consensus weights on both datasets
            train_f1 = evaluate_weights(X_scaled_train, y_train, consensus_weights)
            val_f1 = evaluate_weights(X_scaled_validate, y_validate, consensus_weights)
            
            print(f"\n✅ Consensus Results ({mode}, λ={penalty}):")
            print(f"Training F1: {train_f1:.4f}")
            print(f"Validation F1: {val_f1:.4f}")
            print(f"Generalization Gap: {abs(train_f1 - val_f1):.4f}")
            print("Consensus Weights:", consensus_weights)
            
            results.append({
                'mode': mode,
                'penalty': penalty,
                'weights': consensus_weights,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'gap': abs(train_f1 - val_f1),
                'min_f1': min(train_f1, val_f1)
            })
    
    # Find the most robust result (good performance with small gap)
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    
    for result in results:
        print(f"{result['mode']:12} λ={result['penalty']:.1f}: Train={result['train_f1']:.4f}, "
              f"Val={result['val_f1']:.4f}, Gap={result['gap']:.4f}, Min={result['min_f1']:.4f}")
    
    # Select best result based on minimum F1 score (most conservative)
    best_result = max(results, key=lambda x: x['min_f1'])

    print(f"\n🏆 MOST ROBUST RESULT:")
    print(f"Method: {best_result['mode']} with penalty_lambda={best_result['penalty']}")
    print(f"Training F1: {best_result['train_f1']:.4f}")
    print(f"Validation F1: {best_result['val_f1']:.4f}")
    print(f"Generalization Gap: {best_result['gap']:.4f}")
    print("Robust Weights:", best_result['weights'])