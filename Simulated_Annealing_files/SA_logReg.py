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
    logits = np.dot(X, weights)
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


def simulated_annealing(weights, X, y, parameters,
                       threshold=64, penalty_lambda=0.01, n_bootstrap=100,  # Reduced penalty_lambda
                       initial_temp=100.0, final_temp=0.01, cooling_rate=0.95,
                       max_iterations=1000, neighborhood_size=5):
    """
    Simulated Annealing for weight optimization
    NOTE: This version assumes we want to MAXIMIZE fitness (AUC)
    """
    
    # Initialize
    current_weights = np.array(weights).copy()
    current_fitness = fitness(weights, X, y, penalty_lambda=0.1, n_bootstrap=50)
    
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
        neighbor_fitness =fitness(neighbor_weights, X, y, penalty_lambda=0.1, n_bootstrap=50)
        
        # Calculate acceptance probability
        delta = neighbor_fitness - current_fitness
        
        if delta > 0:  # Better solution (CHANGED: higher AUC is better)
            accept = True
            acceptance_prob = 1.0
        else:  # Worse solution
            acceptance_prob = np.exp(delta / temperature)  # CHANGED: removed negative sign
            accept = np.random.rand() < acceptance_prob
        
        # Accept or reject
        if accept:
            current_weights = neighbor_weights
            current_fitness = neighbor_fitness
            
            # Update best if necessary (CHANGED: higher is better)
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

def adaptive_simulated_annealing(weights, X, y, parameters, penalty_lambda=0.01, n_bootstrap=100,  # Reduced penalty_lambda
                                initial_temp=100.0, final_temp=0.01, max_iterations=1000):
    """
    Adaptive SA that adjusts cooling rate based on acceptance ratio
    NOTE: This version assumes we want to MAXIMIZE fitness (AUC)
    """
    current_weights = np.array(weights).copy()
    current_fitness = fitness(current_weights, X, y, penalty_lambda=0.1, n_bootstrap=100)
    
    best_weights = current_weights.copy()
    best_fitness = current_fitness
    
    temperature = initial_temp
    cooling_rate = 0.95
    
    # Adaptive parameters
    acceptance_window = 50  # Window to calculate acceptance rate
    recent_acceptances = []
    target_acceptance_rate = 0.3  # Target 30% acceptance
    
    fitness_history = []
    temperature_history = []
    
    iteration = 0
    
    while temperature > final_temp and iteration < max_iterations:
        neighbor_weights = generate_neighbor(current_weights, 5, temperature, initial_temp)
        neighbor_fitness =fitness(neighbor_weights, X, y, penalty_lambda=0.1, n_bootstrap=100)
        
        delta = neighbor_fitness - current_fitness
        
        if delta > 0:  # Better solution (CHANGED: higher AUC is better)
            accept = True
        else:
            acceptance_prob = np.exp(delta / temperature)  # CHANGED: removed negative sign
            accept = np.random.rand() < acceptance_prob
        
        recent_acceptances.append(1 if accept else 0)
        if len(recent_acceptances) > acceptance_window:
            recent_acceptances.pop(0)
        
        if accept:
            current_weights = neighbor_weights
            current_fitness = neighbor_fitness
            
            # CHANGED: higher is better
            if current_fitness > best_fitness:
                best_weights = current_weights.copy()
                best_fitness = current_fitness
                print(f"New best at iteration {iteration}: {best_fitness:.6f}")
        
        # Adaptive cooling rate
        if len(recent_acceptances) == acceptance_window:
            acceptance_rate = np.mean(recent_acceptances)
            if acceptance_rate > target_acceptance_rate * 1.2:
                cooling_rate = 0.98  # Cool slower if accepting too much
            elif acceptance_rate < target_acceptance_rate * 0.8:
                cooling_rate = 0.92  # Cool faster if accepting too little
            else:
                cooling_rate = 0.95  # Default rate
        
        fitness_history.append(current_fitness)
        temperature_history.append(temperature)
        
        if iteration % 50 == 0:
            acc_rate = np.mean(recent_acceptances) if recent_acceptances else 0
            print(f"Iteration {iteration}, Temp: {temperature:.4f}, "
                  f"Current: {current_fitness:.6f}, Best: {best_fitness:.6f}, "
                  f"Accept Rate: {acc_rate:.3f}, Cool Rate: {cooling_rate:.3f}")
        
        temperature *= cooling_rate
        iteration += 1
    
    return {
        'best_weights': best_weights,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'temperature_history': temperature_history,
        'iterations': iteration
    }


def run_simulated_annealing(X, y, parameters, empirical_weights):
    """Example of how to run SA"""
    
    print("Running Standard Simulated Annealing...")
    results_standard = simulated_annealing(
        weights=empirical_weights,
        X=X,
        y=y,
        parameters=parameters,
        penalty_lambda=0.01,
        n_bootstrap=100,
        initial_temp=200.0,
        final_temp=0.1,
        cooling_rate=0.95,
        max_iterations=1000,
        neighborhood_size=5
    )
    
    print("\nRunning Adaptive Simulated Annealing...")
    results_adaptive = adaptive_simulated_annealing(
        weights=empirical_weights,
        X=X,
        y=y,
        parameters=parameters,
        penalty_lambda=0.01, 
        n_bootstrap=100,
        initial_temp=200.0,
        final_temp=0.1,
        max_iterations=1000
    )
    
    return results_standard, results_adaptive

if __name__ == "__main__":
    paths = {}
    paths['ZM'] = "/1TB/wYr_model/wYr_datasets/training_ZM.xlsx"
    paths['Questionnaires'] = "/1TB/wYr_model/wYr_datasets/training_questionnaire.xlsx"
    paths['Prospective'] = "/1TB/wYr_model/wYr_datasets/training_followup.xlsx"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    target = f_generate_TARGET_dataset(paths, 4)
    df_base = target.dataset.copy()
    df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()

    feature_cols = ["Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", 
                              "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", 
                              "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed", "ICONFES_Score_Adjusted", "IPAQ_Cat", "Age", "Fall_2"]
    X = df_base[feature_cols].values
    y = df_base["label"].values
    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    empirical_weights = [0.13113595,  0.12291758,  0.23433388, -0.08292311,  0.16714466, -0.47159192,
  -0.05774576, -0.02259974, -0.03646138,  0.23757977,  0.05706306,  0.13853639,
   0.05791303, -0.08366833,  0.0327659,   0.30539868]
    
    results_standard, results_adaptive = run_simulated_annealing(X_scaled, y, parameters, empirical_weights)

    print(f"\nStandard SA - Best Fitness: {results_standard['best_fitness']:.6f}")
    print(f"Adaptive SA - Best Fitness: {results_adaptive['best_fitness']:.6f}")

    print("\nBest Weights from Standard SA:")
    for p, w in zip(parameters, results_standard['best_weights']):
        print(f"{p}: {w}")

    print("\nBest Weights from Adaptive SA:")
    for p, w in zip(parameters, results_adaptive['best_weights']):
        print(f"{p}: {w}")


    
    