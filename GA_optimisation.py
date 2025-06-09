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


def genPopulation(size, empirical_weights):
    population = [empirical_weights.copy()]
    for _ in range(size - 1):
        new_individual = empirical_weights + np.random.randint(-4, 5, size=len(empirical_weights))
        new_individual = np.clip(new_individual, 0, 100)
        population.append(new_individual)
    return np.array(population)

def selectBest(population, fitness_scores, num_best=20):
    best_indices = np.argsort(fitness_scores)[:num_best]
    return population[best_indices]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutation(individual, mutation_rate=0.3):
    mutated = individual.copy()
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.clip(individual[i] + np.random.randint(-2, 3), 0, 100)
    return mutated

def tournament_selection(population, fitness_scores, num_selected=60, tournament_size=5):
    """
    Select individuals using tournament selection.

    Args:
        population (np.ndarray): Current population.
        fitness_scores (np.ndarray): Fitness values for each individual.
        num_selected (int): How many individuals to select.
        tournament_size (int): Number of individuals per tournament.

    Returns:
        np.ndarray: Selected population (num_selected, n_genes)
    """
    selected = []
    for _ in range(num_selected):
        participants = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = participants[np.argmin(fitness_scores[participants])]
        selected.append(population[best_idx])
    return np.array(selected)


def genetic_algorithm(initial_population, population_size, generations, mutation_rate, df_base, parameters, mean_vals, std_vals, dirns, threshold=64):
    population = genPopulation(population_size, initial_population)
    best_individual = None
    best_fitness = float('inf')

    for generation in range(generations):
        fitness_scores = np.array([
            fitness(ind, df_base, parameters, mean_vals, std_vals, dirns, threshold)
            for ind in population
        ])
        best_idx = np.argmin(fitness_scores)
        if fitness_scores[best_idx] < best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_individual = population[best_idx]

        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

        selected = tournament_selection(population, fitness_scores, num_selected=20, tournament_size=5)
        next_generation = []

        while len(next_generation) < len(population):
            parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutation(child1, mutation_rate))
            next_generation.append(mutation(child2, mutation_rate))

        population = np.array(next_generation[:len(population)])

    return best_individual, best_fitness





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
    mean_vals = {k: float(g["Mean/cutoff"]) for k, g in df_meta.groupby("Parameter")}
    std_vals = {k: float(g["StdDev"]) for k, g in df_meta.groupby("Parameter")}
    dirns = {k: str(g["Faller_if"].to_list()[0]) for k, g in df_meta.groupby("Parameter")}
    empirical_weights = df_meta["Weights"].values.tolist()

    best_weights, best_fitness = genetic_algorithm(
        initial_population=empirical_weights,
        population_size=100,
        generations=400,
        mutation_rate=0.2,
        df_base=df_base,
        parameters=parameters,
        mean_vals=mean_vals,
        std_vals=std_vals,
        dirns=dirns,
        threshold=64
    )

    print("Best Weights:", best_weights)
    print("Best Fitness:", best_fitness)

