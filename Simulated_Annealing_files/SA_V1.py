import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, '/1TB/wYr_model')
sys.path.insert(0, '/1TB/wYr_model/GA_optimisation_files') 

from wYr_model_original import f_generate_TARGET_dataset

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
    f1s = []

    for i in range(n_bootstrap):
        if i % 50 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")

        df_sample = resample(df_base, replace=True, n_samples=150, random_state=42*i)
        
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
    fitness_score = mean_f1 - penalty_lambda * std_f1

    return fitness_score

def simulated_annealing(initial_weights, df_base, parameters, mean_vals, std_vals, dirns,
                       threshold=64, penalty_lambda=0.01, n_bootstrap=100,  # Reduced penalty_lambda
                       initial_temp=100.0, final_temp=0.01, cooling_rate=0.95,
                       max_iterations=1000, neighborhood_size=5):
    """
    Simulated Annealing for weight optimization
    NOTE: This version assumes we want to MAXIMIZE fitness (AUC)
    """
    
    # Initialize
    current_weights = np.array(initial_weights).copy()
    current_fitness = fitness(current_weights, df_base, parameters, mean_vals, 
                             std_vals, dirns, threshold, penalty_lambda=penalty_lambda, 
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
        neighbor_fitness = fitness(neighbor_weights, df_base, parameters, mean_vals, 
                                 std_vals, dirns, threshold, penalty_lambda=penalty_lambda, 
                                 n_bootstrap=n_bootstrap)
        
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
    
    # Adaptive neighborhood size based on temperature
    temp_ratio = temperature / initial_temp
    adaptive_size = max(1, int(neighborhood_size * temp_ratio))
    
    # Method 1: Change random subset of weights
    num_changes = np.random.randint(1, min(len(current_weights), adaptive_size) + 1)
    indices_to_change = np.random.choice(len(current_weights), num_changes, replace=False)
    
    for idx in indices_to_change:
        # Larger changes when temperature is high
        max_change = max(1, int(neighborhood_size * temp_ratio))
        change = np.random.randint(-max_change, max_change + 1)
        neighbor[idx] = np.clip(current_weights[idx] + change, 0, 100)
    
    return neighbor

def generate_neighbor_gaussian(current_weights, neighborhood_size, temperature, initial_temp):
    """Alternative neighbor generation using Gaussian perturbation"""
    neighbor = current_weights.copy()
    
    # Temperature-dependent standard deviation
    temp_ratio = temperature / initial_temp
    std_dev = neighborhood_size * temp_ratio
    
    # Add Gaussian noise to all weights
    noise = np.random.normal(0, std_dev, len(current_weights))
    neighbor = current_weights + noise
    neighbor = np.clip(neighbor, 0, 100).astype(int)
    
    return neighbor

def adaptive_simulated_annealing(initial_weights, df_base, parameters, mean_vals, std_vals, dirns,
                                threshold=64, penalty_lambda=0.01, n_bootstrap=100,  # Reduced penalty_lambda
                                initial_temp=100.0, final_temp=0.01, max_iterations=1000):
    """
    Adaptive SA that adjusts cooling rate based on acceptance ratio
    NOTE: This version assumes we want to MAXIMIZE fitness (AUC)
    """
    current_weights = np.array(initial_weights).copy()
    current_fitness = fitness(current_weights, df_base, parameters, mean_vals, 
                             std_vals, dirns, threshold, penalty_lambda=penalty_lambda, 
                             n_bootstrap=n_bootstrap)
    
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
        neighbor_fitness = fitness(neighbor_weights, df_base, parameters, mean_vals, 
                                 std_vals, dirns, threshold, penalty_lambda=penalty_lambda, 
                                 n_bootstrap=n_bootstrap)
        
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

def plot_sa_results(results):
    """Plot SA optimization results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Fitness over time
    axes[0,0].plot(results['fitness_history'])
    axes[0,0].set_title('Fitness Over Time')
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Fitness')
    axes[0,0].grid(True)
    
    # Temperature over time
    axes[0,1].plot(results['temperature_history'])
    axes[0,1].set_title('Temperature Over Time')
    axes[0,1].set_xlabel('Iteration')
    axes[0,1].set_ylabel('Temperature')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True)
    
    # Acceptance probability over time
    if 'acceptance_history' in results:
        axes[1,0].plot(results['acceptance_history'])
        axes[1,0].set_title('Acceptance Probability Over Time')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Acceptance Probability')
        axes[1,0].grid(True)
    
    # Best weights visualization
    axes[1,1].bar(range(len(results['best_weights'])), results['best_weights'])
    axes[1,1].set_title('Best Weights Found')
    axes[1,1].set_xlabel('Parameter Index')
    axes[1,1].set_ylabel('Weight Value')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage function
def run_simulated_annealing(df_base, parameters, mean_vals, std_vals, dirns, empirical_weights):
    """Example of how to run SA"""
    
    print("Running Standard Simulated Annealing...")
    results_standard = simulated_annealing(
        initial_weights=empirical_weights,
        df_base=df_base,
        parameters=parameters,
        mean_vals=mean_vals,
        std_vals=std_vals,
        dirns=dirns,
        threshold=64,
        penalty_lambda=0.01,
        n_bootstrap=100,
        initial_temp=200.0,
        final_temp=0.1,
        cooling_rate=0.95,
        max_iterations=500,
        neighborhood_size=5
    )
    
    print("\nRunning Adaptive Simulated Annealing...")
    results_adaptive = adaptive_simulated_annealing(
        initial_weights=empirical_weights,
        df_base=df_base,
        parameters=parameters,
        mean_vals=mean_vals,
        std_vals=std_vals,
        dirns=dirns,
        threshold=64,
        penalty_lambda=0.01, 
        n_bootstrap=100,
        initial_temp=200.0,
        final_temp=0.1,
        max_iterations=500
    )
    
    return results_standard, results_adaptive


if __name__ == "__main__":
    paths = {}
    paths['ZM'] = "/1TB/Dataset_Feb2025/Training_ZM.xlsx"
    paths['Questionnaires'] = "/1TB/Dataset_Feb2025/Training_questionnaire.xlsx"
    paths['Prospective'] = "/1TB/Dataset_Feb2025/Training_followup.xlsx"
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

results_standard, results_adaptive = run_simulated_annealing(df_base, parameters, mean_vals, std_vals, dirns, empirical_weights)

print(f"\nStandard SA - Best Fitness: {results_standard['best_fitness']:.6f}")
print(f"Adaptive SA - Best Fitness: {results_adaptive['best_fitness']:.6f}")

print("\nBest Weights from Standard SA:")
for p, w in zip(parameters, results_standard['best_weights']):
    print(f"{p}: {w}")

print("\nBest Weights from Adaptive SA:")
for p, w in zip(parameters, results_adaptive['best_weights']):
    print(f"{p}: {w}")


plot_sa_results(results_standard)
plot_sa_results(results_adaptive)
