import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import f1_score
from scipy.special import expit
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import ast  # Add this import for safe string evaluation

sys.path.insert(0, '/1TB/wYr_model') 

from wYr_model_original import f_generate_TARGET_dataset

def parse_weights(weights_value):
    """Parse weights from Excel - handle both string and numeric formats"""
    if isinstance(weights_value, str):
        try:
            # Try to parse as a string representation of a list
            if weights_value.startswith('[') and weights_value.endswith(']'):
                return ast.literal_eval(weights_value)
            else:
                # Try to parse as comma-separated values
                return [float(x.strip()) for x in weights_value.split(',')]
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse weights string: {weights_value}")
            return None
    elif isinstance(weights_value, (list, np.ndarray)):
        return weights_value
    else:
        # Single numeric value
        return [float(weights_value)]

def evaluate_weights(X, y, weights):
    """Evaluate weights using logistic regression approach"""
    try:
        weights_array = np.array([float(w) for w in weights])
        
        # Check dimension compatibility
        if len(weights_array) != X.shape[1]:
            print(f"Dimension mismatch: X has {X.shape[1]} features, weights has {len(weights_array)} elements")
            return 0.0
            
        weights_scaled = weights_array / 100
        logits = np.dot(X, weights_scaled)
        probs = expit(logits)
        preds = (probs >= 0.5).astype(int)
        
        # Check for valid predictions
        if len(np.unique(preds)) == 1:
            # All predictions are the same - this will cause F1 issues
            return 0.0
        
        f1 = f1_score(y, preds)
        return f1
    except Exception as e:
        print(f"Error in evaluate_weights: {e}")
        return 0.0

def fitness(weights, X, y, penalty_lambda=0.1, n_bootstrap=100):
    """Fitness function using bootstrap sampling"""
    f1s = []
    debug_info = {
        'failed_evaluations': 0,
        'successful_evaluations': 0,
        'zero_f1_count': 0,
        'sample_predictions': [],
        'sample_f1s': []
    }

    for i in range(n_bootstrap):
        if i % 10 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")

        # Ensure we have enough samples and balanced classes if possible
        sample_size = min(250, len(X))
        X_sample, y_sample = resample(X, y, replace=True, n_samples=sample_size, random_state=42*i)
        
        try:
            f1 = evaluate_weights(X_sample, y_sample, weights)
            if f1 is not None and not np.isnan(f1):
                f1s.append(f1)
                debug_info['successful_evaluations'] += 1
                if f1 == 0.0:
                    debug_info['zero_f1_count'] += 1
                
                # Store some debug info for first few iterations
                if i < 3:
                    debug_info['sample_f1s'].append(f1)
                    # Get predictions for debugging
                    weights_array = np.array([float(w) for w in weights])
                    
                    # Check dimension compatibility before proceeding
                    if len(weights_array) == X_sample.shape[1]:
                        weights_scaled = weights_array / 100
                        logits = np.dot(X_sample, weights_scaled)
                        probs = expit(logits)
                        preds = (probs >= 0.5).astype(int)
                        debug_info['sample_predictions'].append({
                            'unique_preds': np.unique(preds),
                            'unique_labels': np.unique(y_sample),
                            'logits_range': [np.min(logits), np.max(logits)],
                            'probs_range': [np.min(probs), np.max(probs)]
                        })
            else:
                debug_info['failed_evaluations'] += 1
        except Exception as e:
            debug_info['failed_evaluations'] += 1
            continue
        
    if len(f1s) == 0:
        print(f"  WARNING: No valid F1 scores obtained! Debug info: {debug_info}")
        return 0.0
    
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    fitness_score = mean_f1 - penalty_lambda * std_f1

    # Print debug info for first individual of each generation
    if len(debug_info['sample_f1s']) > 0:
        print(f"  Debug: {debug_info['successful_evaluations']}/{n_bootstrap} successful, "
              f"zero F1s: {debug_info['zero_f1_count']}, "
              f"mean F1: {mean_f1:.4f}, sample F1s: {debug_info['sample_f1s']}")
        
        # Print prediction info for first sample
        if debug_info['sample_predictions']:
            pred_info = debug_info['sample_predictions'][0]
            print(f"  Sample predictions: {pred_info['unique_preds']}, "
                  f"labels: {pred_info['unique_labels']}, "
                  f"logits range: [{pred_info['logits_range'][0]:.3f}, {pred_info['logits_range'][1]:.3f}]")

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

def genPopulation(size, num_features, empirical_weights=None):
    """Generate initial population with correct number of features"""
    population = []
    
    # If empirical weights are provided, use them as the first individual
    if empirical_weights is not None and len(empirical_weights) == num_features:
        population.append(np.array(empirical_weights).copy())
        size -= 1  # Reduce size by 1 since we added the empirical weights
    
    # Generate remaining individuals randomly
    for _ in range(size):
        new_individual = np.random.randint(0, 101, size=num_features)
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

def inject_immigrants(num_features, num_immigrants=10):
    """Inject random individuals to maintain diversity"""
    immigrants = []
    for _ in range(num_immigrants):
        immigrant = np.random.randint(0, 101, size=num_features)
        immigrant = np.clip(immigrant, 0, 100)
        immigrants.append(immigrant)
    
    return np.array(immigrants)

def genetic_algorithm_improved(num_features, population_size, generations, base_mutation_rate, 
                             X, y, parameters, penalty_lambda=0.1, n_bootstrap=100, 
                             empirical_weights=None):
    """
    Improved genetic algorithm with updated fitness function
    
    Args:
        num_features: Number of features in the dataset
        population_size: Size of the population
        generations: Number of generations
        base_mutation_rate: Base mutation rate
        X: Feature matrix
        y: Target labels
        parameters: List of parameter names
        penalty_lambda: Penalty for standard deviation in fitness
        n_bootstrap: Number of bootstrap samples
        empirical_weights: Optional empirical weights to seed population
    """
    
    # Generate initial population with correct dimensions
    population = genPopulation(population_size, num_features, empirical_weights)
    
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
                fitness_cache[key] = fitness(ind, X, y, penalty_lambda=penalty_lambda, n_bootstrap=n_bootstrap)
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
        if diversity < 3.0 or stagnation_counter > 50:
            immigrants = inject_immigrants(num_features, num_immigrants=5)
            next_generation.extend(immigrants)
            print(f"  -> Injected immigrants due to {'low diversity' if diversity < 3.0 else 'stagnation'}")
            # Reset stagnation counter partially when injecting immigrants
            if stagnation_counter > 30:
                stagnation_counter = max(0, stagnation_counter - 10)
        
        # Fill remaining slots randomly
        while len(next_generation) < population_size:
            random_individual = np.random.randint(0, 101, size=num_features)
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
                population[i] = np.random.randint(0, 101, size=num_features)
            stagnation_counter = 0
        
    return best_individual, best_fitness, fitness_history, diversity_history

# Usage with updated function signature
if __name__ == "__main__":
    paths = {}
    paths['ZM'] = "/1TB/wYr_model/wYr_datasets/training_ZM.xlsx"
    paths['Questionnaires'] = "/1TB/wYr_model/wYr_datasets/training_questionnaire.xlsx"
    paths['Prospective'] = "/1TB/wYr_model/wYr_datasets/training_followup.xlsx"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    target = f_generate_TARGET_dataset(paths, 4)
    df_base = target.dataset.copy()

    # Get metadata for feature selection
    df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()
    empirical_weights = df_meta["Weights"].values.tolist()

    print(f"Parameters from Excel: {parameters}")
    print(f"Number of parameters: {len(parameters)}")
    print(f"Empirical weights: {empirical_weights}")

    # Select only the features corresponding to the parameters
    all_feature_cols = [col for col in df_base.columns if col not in ["label", "Participant"]]
    print(f"Total features in dataset: {len(all_feature_cols)}")
    
    # Find the intersection of parameters and available features
    selected_features = []
    missing_features = []
    
    for param in parameters:
        if param in all_feature_cols:
            selected_features.append(param)
        else:
            missing_features.append(param)
            print(f"Warning: Parameter '{param}' not found in dataset features")
    
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    if missing_features:
        print(f"Missing features ({len(missing_features)}): {missing_features}")
    
    # Use only the selected features
    if len(selected_features) == 0:
        print("ERROR: No matching features found!")
        sys.exit(1)
    
    # Extract data for selected features only
    X = df_base[selected_features].values
    y = df_base["label"].values

    print(f"Selected dataset shape: {X.shape}")
    print(f"Number of selected features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

    # Adjust empirical weights to match selected features
    if len(selected_features) != len(empirical_weights):
        print(f"Adjusting empirical weights from {len(empirical_weights)} to {len(selected_features)} features")
        # Only use weights for features that were found
        adjusted_weights = []
        for i, param in enumerate(parameters):
            if param in selected_features:
                adjusted_weights.append(empirical_weights[i])
        empirical_weights = adjusted_weights
    
    print(f"Final empirical weights ({len(empirical_weights)}): {empirical_weights}")

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    best_weights, best_fitness, fitness_history, diversity_history = genetic_algorithm_improved(
        num_features=X.shape[1],  # Use number of selected features
        population_size=200,  
        generations=200,
        base_mutation_rate=0.15,  
        X=X,
        y=y,
        parameters=selected_features,  # Use selected features as parameters
        penalty_lambda=0.1, 
        n_bootstrap=50,
        empirical_weights=empirical_weights
    )

    print("Best Weights:", best_weights)
    print("Best Fitness:", best_fitness)
    print("Feature mapping:")
    for i, (feature, weight) in enumerate(zip(selected_features, best_weights)):
        print(f"  {feature}: {weight}")