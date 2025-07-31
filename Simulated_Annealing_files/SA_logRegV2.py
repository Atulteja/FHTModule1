import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.special import expit

sys.path.insert(0, '/1TB/wYr_model')
sys.path.insert(0, '/1TB/wYr_model/GA_optimisation_files') 

from wYr_model_original import f_generate_TARGET_dataset

def evaluate_weights(X, y, weights):
    """Evaluate weights using logistic regression approach"""
    weights_array = np.array([float(w) for w in weights])
    logits = np.dot(X, weights_array)
    probs = expit(logits)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(y, preds)
    return f1

def fitness(weights, X, y, penalty_lambda=0.1, n_bootstrap=100):
    """Fitness function using bootstrap sampling"""
    f1s = []

    for i in range(n_bootstrap):
        if i % 100 == 0:
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
    fitness_score = mean_f1 - penalty_lambda * std_f1

    return max(0, fitness_score)

def generate_neighbor(current_weights, neighborhood_size, temperature, initial_temp):
    """Generate a neighboring solution"""
    neighbor = current_weights.copy()
    seed_percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Adaptive neighborhood size based on temperature
    temp_ratio = temperature / initial_temp
    adaptive_size = max(1, int(neighborhood_size * temp_ratio))
    
    # Method 1: Change random subset of weights
    num_changes = np.random.randint(1, min(len(current_weights), adaptive_size) + 1)
    indices_to_change = np.random.choice(len(current_weights), num_changes, replace=False)
    
    for idx in indices_to_change:
        # Larger changes when temperature is high
        percent = np.random.choice(seed_percentages)
        signs = np.random.choice([-1, 1])
        change = signs * (percent / 100.0) * current_weights[idx]
        neighbor[idx] = current_weights[idx] + change
    return neighbor

def simulated_annealing(initial_weights, X, y, penalty_lambda=0.01, n_bootstrap=50,
                       initial_temp=100.0, final_temp=0.01, cooling_rate=0.95,
                       max_iterations=1000, neighborhood_size=5):
    """
    Simulated Annealing for weight optimization
    NOTE: This version assumes we want to MAXIMIZE fitness (F1)
    """
    
    # Initialize
    current_weights = np.array(initial_weights).copy()
    current_fitness = fitness(current_weights, X, y, penalty_lambda=penalty_lambda, 
                             n_bootstrap=n_bootstrap)
    
    best_weights = current_weights.copy()
    best_fitness = current_fitness
    
    # Tracking
    fitness_history = []
    temperature_history = []
    acceptance_history = []
    
    temperature = initial_temp
    iteration = 0
    
    print(f"Starting SA with initial fitness: {current_fitness:.6f}")
    
    while temperature > final_temp and iteration < max_iterations:
        # Generate neighbor solution
        neighbor_weights = generate_neighbor(current_weights, neighborhood_size, temperature, initial_temp)
        neighbor_fitness = fitness(neighbor_weights, X, y, penalty_lambda=penalty_lambda, 
                                 n_bootstrap=n_bootstrap)
        
        # Calculate acceptance probability
        delta = neighbor_fitness - current_fitness
        
        if delta > 0:  # Better solution (higher F1 is better)
            accept = True
            acceptance_prob = 1.0
        else:  # Worse solution
            acceptance_prob = np.exp(delta / temperature)
            accept = np.random.rand() < acceptance_prob
        
        # Accept or reject
        if accept:
            current_weights = neighbor_weights
            current_fitness = neighbor_fitness
            
            # Update best if necessary (higher is better)
            if current_fitness > best_fitness:
                best_weights = current_weights.copy()
                best_fitness = current_fitness
                print(f"New best at iteration {iteration}: {best_fitness:.6f}")
        
        # Record history
        fitness_history.append(current_fitness)
        temperature_history.append(temperature)
        acceptance_history.append(acceptance_prob)
        
        # Progress reporting
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Temp: {temperature:.4f}, "
                  f"Current: {current_fitness:.6f}, Best: {best_fitness:.6f}, "
                  f"Accept Prob: {acceptance_prob:.4f}")
        
        # Cool down
        temperature *= cooling_rate
        iteration += 1
    
    print(f"SA completed. Best fitness: {best_fitness:.6f}")
    
    return {
        'best_weights': best_weights,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'temperature_history': temperature_history,
        'acceptance_history': acceptance_history,
        'iterations': iteration
    }

def multi_start_simulated_annealing(X, y, parameters, empirical_weights,
                                   n_starts=5, penalty_lambda=0.01, n_bootstrap=50):
    """
    Multi-start SA: Run SA from multiple random starting points
    """
    best_overall_fitness = -np.inf
    best_overall_weights = None
    all_results = []
    
    print(f"Running Multi-start SA with {n_starts} different starting points...")

    seed_percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    for start_idx in range(n_starts):
        print(f"\n--- Starting point {start_idx + 1}/{n_starts} ---")
        # Generate random starting weights (or use empirical for first run)
        if start_idx == 0:
            initial_weights = empirical_weights.copy()
        else:
            percent = np.random.choice(seed_percentages)
            sign = np.random.choice([-1, 1])
            change = sign * (percent / 100.0) * np.array(empirical_weights)
            initial_weights = np.array(empirical_weights) + change
        
        # Run SA from this starting point
        results = simulated_annealing(
            initial_weights=initial_weights,
            X=X,
            y=y,
            penalty_lambda=penalty_lambda,
            n_bootstrap=n_bootstrap,
            initial_temp=150.0,
            final_temp=0.1,
            cooling_rate=0.95,
            max_iterations=300,
            neighborhood_size=5
        )
        
        all_results.append(results)
        
        # Track best overall result
        if results['best_fitness'] > best_overall_fitness:
            best_overall_fitness = results['best_fitness']
            best_overall_weights = results['best_weights'].copy()
            print(f"New global best from start {start_idx + 1}: {best_overall_fitness:.6f}")
    
    return {
        'best_weights': best_overall_weights,
        'best_fitness': best_overall_fitness,
        'all_results': all_results
    }

def jumping_simulated_annealing(initial_weights, X, y, penalty_lambda=0.01, n_bootstrap=50,
                               initial_temp=100.0, final_temp=0.01, cooling_rate=0.95,
                               max_iterations=1000, neighborhood_size=5,
                               jump_threshold=50, min_jump_distance=20, max_jump_distance=50):
    """
    Jumping SA: When stuck in local optima, make strategic jumps
    """
    
    # Initialize
    current_weights = np.array(initial_weights).copy()
    current_fitness = fitness(current_weights, X, y, penalty_lambda=penalty_lambda, 
                             n_bootstrap=n_bootstrap)
    
    best_weights = current_weights.copy()
    best_fitness = current_fitness
    
    # Tracking
    fitness_history = []
    temperature_history = []
    jump_history = []
    
    temperature = initial_temp
    iteration = 0
    stagnation_counter = 0
    last_improvement = 0
    
    print(f"Starting Jumping SA with initial fitness: {current_fitness:.6f}")
    
    while temperature > final_temp and iteration < max_iterations:
        # Generate neighbor solution
        neighbor_weights = generate_neighbor(current_weights, neighborhood_size, temperature, initial_temp)
        neighbor_fitness = fitness(neighbor_weights, X, y, penalty_lambda=penalty_lambda, 
                                 n_bootstrap=n_bootstrap)
        
        # Calculate acceptance probability
        delta = neighbor_fitness - current_fitness
        
        if delta > 0:  # Better solution
            accept = True
            acceptance_prob = 1.0
            stagnation_counter = 0
            last_improvement = iteration
        else:  # Worse solution
            acceptance_prob = np.exp(delta / temperature)
            accept = np.random.rand() < acceptance_prob
            stagnation_counter += 1
        
        # Accept or reject
        if accept:
            current_weights = neighbor_weights
            current_fitness = neighbor_fitness
            
            # Update best if necessary
            if current_fitness > best_fitness:
                best_weights = current_weights.copy()
                best_fitness = current_fitness
                print(f"New best at iteration {iteration}: {best_fitness:.6f}")
        
        # Check if we should make a jump
        should_jump = (stagnation_counter >= jump_threshold and 
                      iteration - last_improvement >= jump_threshold and
                      temperature > final_temp * 10)  # Don't jump near the end
        
        if should_jump:
            print(f"Making strategic jump at iteration {iteration} (stagnation: {stagnation_counter})")
            jump_weights = make_strategic_jump(current_weights, best_weights, 
                                             min_jump_distance, max_jump_distance)
            jump_fitness = fitness(jump_weights, X, y, penalty_lambda=penalty_lambda, 
                                 n_bootstrap=n_bootstrap)
            
            # Always accept the jump (exploration)
            current_weights = jump_weights
            current_fitness = jump_fitness
            stagnation_counter = 0
            jump_history.append(iteration)
            
            # Update best if jump found something better
            if current_fitness > best_fitness:
                best_weights = current_weights.copy()
                best_fitness = current_fitness
                print(f"Jump found new best: {best_fitness:.6f}")
        
        # Record history
        fitness_history.append(current_fitness)
        temperature_history.append(temperature)
        
        # Progress reporting
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Temp: {temperature:.4f}, "
                  f"Current: {current_fitness:.6f}, Best: {best_fitness:.6f}, "
                  f"Stagnation: {stagnation_counter}")
        
        # Cool down
        temperature *= cooling_rate
        iteration += 1
    
    print(f"Jumping SA completed. Best fitness: {best_fitness:.6f}")
    print(f"Made {len(jump_history)} strategic jumps at iterations: {jump_history}")
    
    return {
        'best_weights': best_weights,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'temperature_history': temperature_history,
        'jump_history': jump_history,
        'iterations': iteration
    }

def make_strategic_jump(current_weights, best_weights, min_distance, max_distance):
    """
    Make a strategic jump to a distant point in the solution space
    """
    seed_percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    jump_weights = current_weights.copy()
    
    # Calculate how far we want to jump
    percent = np.random.choice(seed_percentages)
    signs = np.random.choice([-1, 1])
    change = signs * (percent / 100.0) * (max_distance - min_distance)
    jump_distance = (max_distance - min_distance) + change
    
    # Decide how many weights to change (at least half for a significant jump)
    num_changes = np.random.randint(len(current_weights) // 2, len(current_weights))
    indices_to_change = np.random.choice(len(current_weights), num_changes, replace=False)
    
    for idx in indices_to_change:
        # Make a large random change
        if np.random.rand() < 0.3:  # 30% chance to use best_weights as guidance
            direction = np.sign(best_weights[idx] - current_weights[idx])
            percent = np.random.choice(seed_percentages)
            change = direction * np.random.uniform(jump_distance / 2, jump_distance)
        else:
            change = np.random.uniform(-jump_distance, jump_distance)
        
        jump_weights[idx] = current_weights[idx] + change
    
    return jump_weights

def basin_hopping_sa(initial_weights, X, y, penalty_lambda=0.01, n_bootstrap=50,
                    n_basins=10, sa_iterations=200):
    """
    Basin-hopping approach: Run short SA cycles with jumps between basins
    """
    
    best_overall_fitness = -np.inf
    best_overall_weights = None
    basin_history = []
    
    current_weights = np.array(initial_weights).copy()
    
    print(f"Starting Basin-hopping SA with {n_basins} basins...")
    
    for basin_idx in range(n_basins):
        print(f"\n--- Exploring basin {basin_idx + 1}/{n_basins} ---")
        
        # Run SA in current basin
        results = simulated_annealing(
            initial_weights=current_weights,
            X=X,
            y=y,
            penalty_lambda=penalty_lambda,
            n_bootstrap=n_bootstrap,
            initial_temp=100.0,
            final_temp=1.0,
            cooling_rate=0.95,
            max_iterations=sa_iterations,
            neighborhood_size=5
        )
        
        basin_history.append(results)
        
        # Update global best
        if results['best_fitness'] > best_overall_fitness:
            best_overall_fitness = results['best_fitness']
            best_overall_weights = results['best_weights'].copy()
            print(f"New global best in basin {basin_idx + 1}: {best_overall_fitness:.6f}")
        
        # Jump to next basin (except for last iteration)
        if basin_idx < n_basins - 1:
            # Make a strategic jump from current local optimum
            jump_distance = np.random.randint(15, 40) * np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) * 1/100
            current_weights = make_strategic_jump(results['best_weights'], 
                                                best_overall_weights, 
                                                jump_distance, jump_distance + 0.1)
    
    return {
        'best_weights': best_overall_weights,
        'best_fitness': best_overall_fitness,
        'basin_history': basin_history
    }

def simulated_annealing_with_patience(initial_weights, X, y, penalty_lambda=0.01, n_bootstrap=50,
                                     patience=100, initial_temp=100.0, final_temp=0.01):
    """
    SA with early stopping based on patience
    """
    current_weights = np.array(initial_weights).copy()
    current_fitness = fitness(current_weights, X, y, penalty_lambda=penalty_lambda, 
                             n_bootstrap=n_bootstrap)
    
    best_weights = current_weights.copy()
    best_fitness = current_fitness
    
    temperature = initial_temp
    iteration = 0
    patience_counter = 0
    
    fitness_history = []
    
    while temperature > final_temp and patience_counter < patience:
        neighbor_weights = generate_neighbor(current_weights, 5, temperature, initial_temp)
        neighbor_fitness = fitness(neighbor_weights, X, y, penalty_lambda=penalty_lambda, 
                                 n_bootstrap=n_bootstrap)
        
        delta = neighbor_fitness - current_fitness
        
        if delta > 0:
            accept = True
            patience_counter = 0  # Reset patience on improvement
        else:
            acceptance_prob = np.exp(delta / temperature)
            accept = np.random.rand() < acceptance_prob
            patience_counter += 1
        
        if accept:
            current_weights = neighbor_weights
            current_fitness = neighbor_fitness
            
            if current_fitness > best_fitness:
                best_weights = current_weights.copy()
                best_fitness = current_fitness
        
        fitness_history.append(current_fitness)
        temperature *= 0.95
        iteration += 1
    
    return {
        'best_weights': best_weights,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'iterations': iteration,
        'stopped_early': patience_counter >= patience
    }

def adaptive_restart_sa(initial_weights, X, y, penalty_lambda=0.01, n_bootstrap=50,
                       max_restarts=5, patience=100):
    """
    Adaptive restart: Monitor progress and restart when stuck
    """
    
    best_overall_fitness = -np.inf
    best_overall_weights = None
    restart_history = []
    
    for restart_idx in range(max_restarts):
        print(f"\n--- Restart {restart_idx + 1}/{max_restarts} ---")
        
        # Use best weights as starting point after first restart
        if restart_idx == 0:
            start_weights = initial_weights.copy()
        else:
            # Start from best known solution with some perturbation
            start_weights = best_overall_weights.copy()
            # Add some noise to escape local minimum
            noise = np.random.normal(0, 0.1, len(start_weights))
            start_weights = start_weights + noise
        
        # Run SA with early stopping
        results = simulated_annealing_with_patience(
            initial_weights=start_weights,
            X=X,
            y=y,
            penalty_lambda=penalty_lambda,
            n_bootstrap=n_bootstrap,
            patience=patience
        )
        
        restart_history.append(results)
        
        # Update global best
        if results['best_fitness'] > best_overall_fitness:
            best_overall_fitness = results['best_fitness']
            best_overall_weights = results['best_weights'].copy()
            print(f"New global best in restart {restart_idx + 1}: {best_overall_fitness:.6f}")
        
        # Early termination if no improvement
        if restart_idx > 0 and results['best_fitness'] <= restart_history[-2]['best_fitness']:
            print(f"No improvement in restart {restart_idx + 1}, considering early termination")
    
    return {
        'best_weights': best_overall_weights,
        'best_fitness': best_overall_fitness,
        'restart_history': restart_history
    }

def run_advanced_sa_comparison(X, y, parameters, empirical_weights):
    """Run comparison of different SA variants"""
    
    print("=== Running Advanced SA Comparison ===\n")
    
    # 1. Multi-start SA
    print("1. Multi-start Simulated Annealing")
    results_multistart = multi_start_simulated_annealing(
        X, y, parameters, empirical_weights
    )
    
    # 2. Jumping SA
    print("\n2. Jumping Simulated Annealing")
    results_jumping = jumping_simulated_annealing(
        empirical_weights, X, y
    )
    
    # 3. Basin-hopping SA
    print("\n3. Basin-hopping Simulated Annealing")
    results_basin = basin_hopping_sa(
        empirical_weights, X, y
    )
    
    # 4. Adaptive restart SA
    print("\n4. Adaptive Restart Simulated Annealing")
    results_restart = adaptive_restart_sa(
        empirical_weights, X, y
    )
    
    # Compare results
    print("\n=== COMPARISON RESULTS ===")
    print(f"Multi-start SA:      {results_multistart['best_fitness']:.6f}")
    print(f"Jumping SA:          {results_jumping['best_fitness']:.6f}")
    print(f"Basin-hopping SA:    {results_basin['best_fitness']:.6f}")
    print(f"Adaptive Restart SA: {results_restart['best_fitness']:.6f}")
    
    return {
        'multistart': results_multistart,
        'jumping': results_jumping,
        'basin': results_basin,
        'restart': results_restart
    }

if __name__ == "__main__":
    paths = {}
    paths['ZM'] = "/1TB/wYr_model/wYr_datasets/training_ZM.xlsx"
    paths['Questionnaires'] = "/1TB/wYr_model/wYr_datasets/training_questionnaire.xlsx"
    paths['Prospective'] = "/1TB/wYr_model/wYr_datasets/training_followup.xlsx"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    target = f_generate_TARGET_dataset(paths, 4)
    df_base = target.dataset.copy()

    feature_cols = ["Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", 
                              "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", 
                              "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed", "ICONFES_Score_Adjusted", "IPAQ_Cat", "Age", "Fall_2"]
    X = df_base[feature_cols].values
    y = df_base["label"].values
    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()
    empirical_weights = [0.13113595,  0.12291758,  0.23433388, -0.08292311,  0.16714466, -0.47159192,
  -0.05774576, -0.02259974, -0.03646138,  0.23757977,  0.05706306,  0.13853639,
   0.05791303, -0.08366833,  0.0327659,   0.30539868]
    
    results = run_advanced_sa_comparison(X_scaled, y, parameters, empirical_weights)

    print("\nBest Weights from Multi-start SA:")
    for p, w in zip(parameters, results['multistart']['best_weights']):
        print(f"{p}: {w}")

    print("\nBest Weights from Jumping SA:")
    for p, w in zip(parameters, results['jumping']['best_weights']):
        print(f"{p}: {w}")

    print("\nBest Weights from Basin-Hopping SA:")
    for p, w in zip(parameters, results['basin']['best_weights']):
        print(f"{p}: {w}")

    print("\nBest Weights from Adaptive SA:")
    for p, w in zip(parameters, results['restart']['best_weights']):
        print(f"{p}: {w}")