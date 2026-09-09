import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("matches.csv")
df["total_goals"] = df["goals_home"] + df["goals_away"]
df.loc[df["decided_by"] == "regulation", "decided_group"] = "regulation"
df.loc[df["decided_by"] != "regulation", "decided_group"] = "extra_time_or_penalties"

knockout = df[df["stage"] == "Knockout"].copy()
N_POPULATION = len(knockout)
print("population (knockout matches):", N_POPULATION)

np.random.seed(3)
sample = knockout.sample(n=30, random_state=3)

reg = sample[sample["decided_group"] == "regulation"]["total_goals"]
et = sample[sample["decided_group"] == "extra_time_or_penalties"]["total_goals"]

print("sample size:", len(sample))
print("regulation matches in sample:", len(reg))
print("extra time/penalties matches in sample:", len(et))

print("\n--- descriptive statistics: total goals per match (sample) ---")
print("overall mean:", sample["total_goals"].mean())
print("overall median:", sample["total_goals"].median())
print("overall std dev:", sample["total_goals"].std())

print("\nregulation - mean:", reg.mean(), "median:", reg.median(), "std:", reg.std())
print("extra time/pens - mean:", et.mean(), "median:", et.median(), "std:", et.std())

n = len(sample)
mean = sample["total_goals"].mean()
sem = stats.sem(sample["total_goals"])
z_critical = stats.norm.ppf(0.975)
margin_of_error = z_critical * sem
ci = (mean - margin_of_error, mean + margin_of_error)
print("\n95% CI for mean goals per match (knockout sample):", ci)

mean_reg = reg.mean()
mean_et = et.mean()
std_reg = reg.std()
std_et = et.std()
n_reg = len(reg)
n_et = len(et)

standard_error_diff = np.sqrt((std_reg ** 2 / n_reg) + (std_et ** 2 / n_et))
t_statistic = (mean_et - mean_reg) / standard_error_diff
degrees_of_freedom = min(n_reg - 1, n_et - 1)
p_value = 2 * stats.t.sf(abs(t_statistic), df=degrees_of_freedom)

print("\ntwo-sample t-test (seperate variances, conservative df)")
print("mean differnce (extra time/pens minus regulation):", mean_et - mean_reg)
print("standard error of differnce:", standard_error_diff)
print("degrees of freedom:", degrees_of_freedom)
print("t statistic:", t_statistic)
print("p value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("reject H0: significant difference in goals between regulation and extra time/penalty matches")
else:
    print("fail to reject H0: no significant difference")
