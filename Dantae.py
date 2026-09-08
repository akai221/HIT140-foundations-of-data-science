import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("matches.csv")
df["total_goals"] = df["goals_home"] + df["goals_away"]
df["decided_group"] = df["decided_by"].apply(
    lambda x: "regulation" if x == "regulation" else "extra_time_or_penalties"
)

knockout = df[df["stage"] == "Knockout"].copy()
N_POPULATION = len(knockout)
print("population (knockout matches):", N_POPULATION)

np.random.seed(3)
sample = knockout.sample(n=25, random_state=3)

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
ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
print("\n95% CI for mean goals per match (knockout sample):", ci)

shapiro_reg = stats.shapiro(reg)
shapiro_et = stats.shapiro(et)
print("\nShapiro-Wilk normality test:")
print("regulation group: W =", shapiro_reg.statistic, "p =", shapiro_reg.pvalue)
print("extra time/pens group: W =", shapiro_et.statistic, "p =", shapiro_et.pvalue)

levene = stats.levene(reg, et)
print("\nLevene's test for equal variances: stat =", levene.statistic, "p =", levene.pvalue)

t_stat, p_value = stats.ttest_ind(reg, et, equal_var=False)
print("\ntwo-sample t-test (Welch's, unequal variances assumed)")
print("t statistic:", t_stat)
print("p value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("reject H0: significant difference in goals between regulation and extra time/penalty matches")
else:
    print("fail to reject H0: no significant difference")

u_stat, u_pvalue = stats.mannwhitneyu(reg, et, alternative="two-sided")
print("\nMann-Whitney U test (non-parametric alternative, robustness check)")
print("U statistic:", u_stat)
print("p value:", u_pvalue)
print("conclusion agrees with t-test:" , (u_pvalue < alpha) == (p_value < alpha))
