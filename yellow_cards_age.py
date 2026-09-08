import pandas as pd
import math
from scipy.stats import ttest_ind, norm
players = pd.read_csv("worldcup2026_playerstats.csv", header=1)
print("Raw player data:", players.shape)
players = players[["Player", "Age", "Min", "90s", "CrdY"]]
print(players.head())
print("Selected player data:", players.shape)
print("Player keys:", players.keys())
print("Before removing missing data:", players.shape)
players = players.dropna()
print("After removing missing data:", players.shape)
players = players[players["Min"] >= 90].copy()
print("After applying the 90-minute requirement:", players.shape)
players["Yellow_Cards_Per_90"] = (players["CrdY"] / players["Min"]) * 90
players["Age_Group"] = pd.cut(
    players["Age"],
    bins=[0, 29, 100],
    labels=["Under 30", "30 and over"])
print(players.head())
print("Final player data:", players.shape)
print("Final player keys:", players.keys())
players.to_csv("cleaned_yellow_cards_age.csv", index=False)
under_30 = players[players["Age_Group"] == "Under 30"]
thirty_and_over = players[players["Age_Group"] == "30 and over"]
under_30_sample = under_30.sample(n=100, random_state=42)
thirty_and_over_sample = thirty_and_over.sample(n=100, random_state=42)
stratified_sample = pd.concat([under_30_sample, thirty_and_over_sample], ignore_index=True)
stratified_sample = stratified_sample.sample(frac=1, random_state=42)
stratified_sample = stratified_sample.reset_index(drop=True)
print("Stratified sample:", stratified_sample.shape)
print(stratified_sample["Age_Group"].value_counts())
stratified_sample.to_csv("stratified_yellow_cards_sample.csv", index=False)

under_30_data = stratified_sample[
    stratified_sample["Age_Group"] == "Under 30"
]["Yellow_Cards_Per_90"]

thirty_and_over_data = stratified_sample[
    stratified_sample["Age_Group"] == "30 and over"
]["Yellow_Cards_Per_90"]

print("\nDescriptive statistics for players under 30:")
print(under_30_data.describe())

print("\nDescriptive statistics for players aged 30 and over:")
print(thirty_and_over_data.describe())

mean_under_30 = under_30_data.mean()
mean_thirty_and_over = thirty_and_over_data.mean()

mean_difference = mean_thirty_and_over - mean_under_30

standard_error_difference = math.sqrt(
    (thirty_and_over_data.std() ** 2 / len(thirty_and_over_data))
    + (under_30_data.std() ** 2 / len(under_30_data))
)

z_critical = norm.ppf(0.975)
margin_of_error = z_critical * standard_error_difference
confidence_interval = (mean_difference - margin_of_error, mean_difference + margin_of_error)

print("\nConfidence interval results:")
print("Mean difference (30 and over minus under 30):", mean_difference)
print("Standard error:", standard_error_difference)
print("95% confidence interval:", confidence_interval)
t_statistic, p_value = ttest_ind(
    thirty_and_over_data,
    under_30_data,
    equal_var=False,
    alternative="greater"
)

alpha = 0.05

print("\nTwo-sample t-test results:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print("Significance level:", alpha)

if p_value < alpha:
    print("Decision: Reject the null hypothesis.")
    print("Conclusion: There is evidence that players aged 30 and over have a higher average yellow-card rate.")
else:
    print("Decision: Fail to reject the null hypothesis.")
    print("Conclusion: There is not enough evidence that players aged 30 and over have a higher average yellow-card rate.")
