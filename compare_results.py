import pandas as pd

def compare_predictions(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Merge the datasets on Participant
    common_cols = [col for col in df1.columns if col in df2.columns and col not in ['Participant']]
    merged = pd.merge(
        df1[['Participant', 'prediction', 'risk score']],
        df2[['Participant', 'prediction', 'risk score']],
        on='Participant',
        suffixes=('_1', '_2'),
        how='inner'
    )
    merged2 = pd.merge(df1, df2, on='Participant', suffixes=('_1', '_2'), how='inner')

    # Add comparison column
    merged['equal'] = merged['prediction_1'] == merged['prediction_2']
    merged2['equal'] = merged2['prediction_1'] == merged2['prediction_2']

    # Print summary
    total = len(merged)
    equal = merged['equal'].sum()
    diff = total - equal
    print(f"Total Participants compared: {total}")
    print(f"Predictions equal: {equal}")
    print(f"Predictions different: {diff}")

    mismatched = merged2[merged2['equal'] == False].copy()

    diff_rows = []
    for _, row in mismatched.iterrows():
        participant_id = row['Participant']
        row_diff = {'Participant': participant_id}
        for col in common_cols:
            val1 = row.get(f'{col}_1', None)
            val2 = row.get(f'{col}_2', None)
            if pd.isna(val1) and pd.isna(val2):
                continue
            if val1 != val2:
                row_diff[f"{col}_1"] = val1
                row_diff[f"{col}_2"] = val2
        diff_rows.append(row_diff)

    diff_df = pd.DataFrame(diff_rows)

    return merged, diff_df

if __name__ == "__main__":
    # Example usage
    df1 = pd.read_csv("/1TB/booster_riskscore_predictions_UPDATED.csv")
    df2 = pd.read_csv("/1TB/Booster_processed_data_25072025.csv")

    result, diff_df = compare_predictions(df1, df2)

    # Save results
    result.to_csv("comparison_results.csv", index=False)
    diff_df.to_csv("differences.csv", index=False)
    print("Comparison saved to comparison_results.csv")
