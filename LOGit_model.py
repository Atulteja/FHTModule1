import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report, roc_curve, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold


from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET

def f_generate_TARGET_dataset(paths, num_followups=4):
    # read ZM data
    zm = ZURICHMOVEDATA(paths['ZM'])
    zm.read_dataset()
    print('zm dataset - shape:', zm.dataset.shape)

    # read questionnaire data
    questionnaire = QUESTIONNAIREDATA(paths['Questionnaires'], features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted'])
    questionnaire.read_dataset()
    print('questionnaire dataset - shape:', questionnaire.dataset.shape)

    missing_info_participants = []
    for participant in zm.dataset["Participant"]:
        if participant not in list(questionnaire.dataset["Participant"]):
            missing_info_participants.append(participant)
    
    if len(missing_info_participants)>0:
        print("Information missing for following participants:")
        for i in range(len(missing_info_participants)):
            print(missing_info_participants[i])

    # construct the dataset (TARGET) - without followup information
    target = DATASET()
    target.merge_datasets(zm.dataset, questionnaire.dataset,
                        {'first_dataset': 'Participant', 'second_dataset': 'Participant'})
    print('TARGET dataset - shape:', target.dataset.shape)
    target.dataset.dropna(inplace=True)
    print('TARGET dataset - shape:', target.dataset.shape)

    if num_followups>0 and len(paths['Prospective'])>0:
        # read prospective data
        followup = PROSPECTIVEDATA(paths['Prospective'])
        followup.read_dataset()
        followup.generate_labels(num_followups=num_followups)

        # construct the dataset (TARGET) - with followup information
        complete_dataset = DATASET()
        complete_dataset.merge_datasets(target.dataset, followup.labels,
                            {'first_dataset': 'Participant', 'second_dataset': 'Participant'})
        complete_dataset.dataset.dropna(inplace=True)

        print("---------------------------")
        print(complete_dataset.dataset.label.value_counts())
        print("---------------------------")

        target = complete_dataset
    else:
        print("Follow up information not incorporated")

    return target

def compute_youdens_index(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens + spec - 1

def generate_labels_features(df):
    feature_cols = ["Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", 
                              "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", 
                              "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed", "ICONFES_Score_Adjusted", "IPAQ_Cat", "Age", "Fall_2"]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"Features: {feature_cols}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y

def standardize_feature(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def dataset_split(X_scaled, y):

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    return X_train, X_test, y_train, y_test

def apply_undersampling(X_train, y_train, random_state=42):
    """
    Apply random undersampling to balance the training set
    """
    print("\n--- Before Undersampling ---")
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    undersampler = RandomUnderSampler(random_state=random_state)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    
    print("\n--- After Undersampling ---")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Original training size: {X_train.shape[0]}")
    print(f"Resampled training size: {X_train_resampled.shape[0]}")
    
    return X_train_resampled, y_train_resampled

def tune_logistic_regression(X_train, y_train, cv=5):
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']
    }

    model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
                               
    
    # best_score = -1
    # best_params = None
    # best_model = None
    
    # print("\n--- Hyperparameter Tuning Results ---")

    # # skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # for C in param_grid['C']:
    #     for fit_intercept in param_grid['fit_intercept']:
    #             # Train model with current parameters
    #             model = logistic_reg( 
    #                 X_train=X_train, 
    #                 y_train=y_train, 
    #                 C=C, 
    #                 fit_intercept=fit_intercept, 
    #             )

    #             cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    #             mean_f1 = np.mean(cv_score)

    #             if mean_f1 > best_score:
    #                 best_score = mean_f1
    #                 best_params = {'C': C, 'fit_intercept': fit_intercept}
    #                 best_model = model
                
    #             print(f"C={C}, fit_intercept={fit_intercept}: F1={mean_f1:.4f}")
        
    print(f"\nBest Params: {grid_search.best_params_}")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")  
    return grid_search.best_estimator_, grid_search.best_params_

def tune_random_forest(X_train, y_train, cv=5):
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }

    model = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        model, 
        param_grid, 
        n_iter=50, 
        cv=cv, 
        scoring='f1', 
        n_jobs=-1, 
        verbose=1, 
        random_state=42
    )
    random_search.fit(X_train, y_train)

    print(f"\nBest Params: {random_search.best_params_}")
    print(f"Best F1 Score: {random_search.best_score_:.4f}")  
    return random_search.best_estimator_, random_search.best_params_

def tune_gradient_boosting(X_train, y_train, cv=5):
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0],
    }

    model = GradientBoostingClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        model, 
        param_grid, 
        n_iter=50, 
        cv=cv, 
        scoring='f1', 
        n_jobs=-1, 
        verbose=1, 
        random_state=42
    )
    random_search.fit(X_train, y_train)

    print(f"\nBest Params: {random_search.best_params_}")
    print(f"Best F1 Score: {random_search.best_score_:.4f}")  
    return random_search.best_estimator_, random_search.best_params_

def tune_svm(X_train, y_train, cv=5):
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto' , 0.001, 0.01, 0.1, 1.0],
        'class_weight': [None, 'balanced']
    }

    model = SVC(probability=True, random_state=42)
    random_search = RandomizedSearchCV(
        model, 
        param_grid, 
        n_iter=50, 
        cv=cv, 
        scoring='f1', 
        n_jobs=-1, 
        verbose=1, 
        random_state=42
    )
    random_search.fit(X_train, y_train)
    print(f"\nBest Params: {random_search.best_params_}")
    print(f"Best F1 Score: {random_search.best_score_:.4f}")
    return random_search.best_estimator_, random_search.best_params_

def ensemble_predictions(models, X_test):
    predictions = []
    probabilities = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = y_pred.astype(int)
        predictions.append(y_pred)
        probabilities.append(y_pred_proba)
        print(f"{name} predictions: {y_pred[:5]}")
        print(f"{name} probabilities: {y_pred_proba[:5]}")

    predictions_array = np.array(predictions)
    ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, arr=predictions_array)

    ensemble_proba = np.mean(probabilities, axis=0)
    return ensemble_pred, ensemble_proba

def find_optimal_threshold(y_true, y_scores):
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    print(f"Optimal threshold: {best_threshold:.4f}, F1 Score at best threshold: {best_f1:.4f}")

    return best_threshold
                         

def logistic_reg(X_train, y_train, C=1.0, fit_intercept=True):
    logit = LogisticRegression(
        penalty='l2', 
        solver='liblinear', 
        C=C, 
        fit_intercept=fit_intercept, 
        random_state=42,
    )
    logit.fit(X_train, y_train)
    print("Model coefficients:", logit.coef_)
    print("Model intercept:", logit.intercept_)
    return logit

def generate_pred(model, X_set):
    y_pred = model.predict(X_set)
    y_pred_proba = model.predict_proba(X_set)[:,1]
    return y_pred, y_pred_proba

def evaluate_predictions(y_true, y_pred, y_pred_proba):
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn+fp)>0 else 0
    sens = tp / (tp + fn) if (tp+fn)>0 else 0
    acc = (tp + tn) / (tn + fp + fn + tp)

    yidx = sens + spec - 1
    plr = sens / (1-spec) if (1-spec)>0 else 0
    f1_ = f1_score(y_true, y_pred)
    aucroc_ = roc_auc_score(y_true, y_pred_proba)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Specificity: {spec:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Youdens Index: {yidx:.4f}")
    print(f"PLR: {plr:.4f}")
    print(f"F1 Score: {f1_:.4f}")
    print(f"AUC-ROC: {aucroc_:.4f}")

    return spec, sens, acc, yidx, plr, f1_, aucroc_

def multiple_algorithms_model(paths, num_followups=4, use_undersampling=True, algorithms=['logistic', 'random_forest', 'gradient_boosting', 'svm']):
    df = f_generate_TARGET_dataset(paths, num_followups=num_followups)
    X, y = generate_labels_features(df.dataset)
    X_scaled = standardize_feature(X)
    X_train, X_test, y_train, y_test = dataset_split(X_scaled, y)

    if use_undersampling:
        X_train, y_train = apply_undersampling(X_train, y_train)

    trained_models = {}
    results = {}

    print(f"\nTraining {len(algorithms)} algorithms...\n")

    for algo in algorithms:
        print(f"{'='*50}")
        print(f"TRAINING {algo.upper().replace('_', ' ')}")
        print(f"{'='*50}")

        try:
            if algo == 'logistic':
                model, best_params = tune_logistic_regression(X_train, y_train)
            elif algo == 'random_forest':
                model, best_params = tune_random_forest(X_train, y_train)
            elif algo == 'gradient_boosting':
                model, best_params = tune_gradient_boosting(X_train, y_train)
            elif algo == 'svm':
                model, best_params = tune_svm(X_train, y_train)
            else:
                print(f"Unknown algorithm: {algo}")
                continue

            trained_models[algo] = model
            print(f"Best Params for {algo}: {best_params}")
            
            print(f"\n---{algo.upper()} Test Set Evaluation---")
            y_pred, y_pred_proba = generate_pred(model, X_test)
            optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            spec, sens, acc, yidx, plr, f1_, aucroc_ = evaluate_predictions(y_test, y_pred_optimal, y_pred_proba)
            results[algo] = {
                'model': model,
                'params': best_params,
                'threshold': optimal_threshold,
                'f1': f1_,
                'spec': spec,
                'sens': sens,
                'yidx': yidx,
                'plr': plr,
                'aucroc': aucroc_,
                'acc': acc
            }
        except Exception as e:
            print(f"Error training {algo}: {e}")
    
    if len(trained_models) > 1:
        print(f"\n{'='*50}")
        print("ENSEMBLE MODEL")
        print(f"{'='*50}")

        try:
            ensemble_pred, ensemble_proba = ensemble_predictions(trained_models, X_test)
            optimal_threshold = find_optimal_threshold(y_test, ensemble_proba)
            ensemble_pred_optimal = (ensemble_proba >= optimal_threshold).astype(int)
            print(f"\n--- ENSEMBLE Test Set Evaluation ---")
            spec, sens, acc, yidx, plr, f1_, aucroc_ = evaluate_predictions(y_test, ensemble_pred_optimal, ensemble_proba)
            results['ensemble'] = {
                'model': 'ensemble',
                'params': None,
                'threshold': optimal_threshold,
                'f1': f1_,
                'spec': spec,
                'sens': sens,
                'yidx': yidx,
                'plr': plr,
                'aucroc': aucroc_,
                'acc': acc
            }
        except Exception as e:
            print(f"Error creating ensemble model: {e}")

    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")

    print(f"{'Model':<20} {'F1':<8} {'Spec':<8} {'Sens':<8} {'Acc':<8} {'YIdx':<8} {'PLR':<8} {'AUC-ROC':<8}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for model_name, metrics in sorted_results:
        print(f"{model_name:<20} {metrics['f1']:<8.4f} {metrics['spec']:<8.4f} {metrics['sens']:<8.4f} "
              f"{metrics['acc']:<8.4f} {metrics['yidx']:<8.4f} {metrics['plr']:<8.4f} {metrics['aucroc']:<8.4f}")
        
    best_model_name, best_metrics = sorted_results[0]
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"F1: {best_metrics['f1']:.4f}, Spec: {best_metrics['spec']:.4f}, "
          f"Sens: {best_metrics['sens']:.4f}, Acc: {best_metrics['acc']:.4f}, F1: {best_metrics['f1']:.4f}, "
          f"YIdx: {best_metrics['yidx']:.4f}, PLR: {best_metrics['plr']:.4f}, AUC-ROC: {best_metrics['aucroc']:.4f}")
    
    return results, trained_models

if __name__ == "__main__":

    paths = {}
    paths['ZM'] = "//1TB/Dataset_Feb2025/TARGETZMParameters_All_20241015.xlsx"
    paths['Questionnaires'] = "/1TB/Dataset_Feb2025/TARGET 5 November 2024 dataset to SEC_051124 numeric n=2291.xlsx"
    paths['Prospective'] = "/1TB/Dataset_Feb2025/TARGET follow-up 18.02.2025 to SEC 26022025 numeric.xls"

    results, models = multiple_algorithms_model(paths, num_followups=4, use_undersampling=True, algorithms=['logistic', 'random_forest', 'gradient_boosting', 'svm'])