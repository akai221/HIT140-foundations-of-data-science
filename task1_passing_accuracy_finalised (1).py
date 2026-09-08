import pandas as pd
from scipy import stats

df = pd.read_csv("passing_accuracy.csv")
data = df["Passing Accuracy"]

mean_val = data.mean()
median_val = data.median()
std_val = data.std()
min_val = data.min()
max_val = data.max()

print("Descriptive Statistics:")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Standard Deviation: {std_val:.2f}")
print(f"Min: {min_val:.2f}")
print(f"Max: {max_val:.2f}")

z_critical = stats.norm.ppf(0.975)
margin_of_error = z_critical * stats.sem(data)
ci_lower = mean_val - margin_of_error
ci_upper = mean_val + margin_of_error

print("\n95% Confidence Interval:")
print((ci_lower, ci_upper))

t_stat, p_val = stats.ttest_1samp(data, 80, alternative="greater")

print("\nT-test against 80%:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
