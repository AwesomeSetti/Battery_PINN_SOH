import pandas as pd
import json

# 1) Capacity table sample (first 10 rows)
cap_path = "data_processed/capacity_table.csv"
df_cap = pd.read_csv(cap_path)

cols = ["traj_id","condition_id","temp_C","cycle","N","capacity","u_true"]
df_cap_sample = df_cap[cols].head(10)

with open("outputs/tables/table_capacity_sample.tex", "w") as f:
    f.write(df_cap_sample.to_latex(index=False, float_format="%.4f"))

print("[OK] Wrote outputs/tables/table_capacity_sample.tex")

# 2) Metrics table (ALL rows = this is important)
met_path = "outputs/tables/metrics_validation.csv"
df_met = pd.read_csv(met_path)

with open("outputs/tables/table_validation_metrics.tex", "w") as f:
    f.write(df_met.to_latex(index=False, float_format="%.4f"))

print("[OK] Wrote outputs/tables/table_validation_metrics.tex")

# 3) Fitted parameters table
params_path = "outputs/tables/fitted_params_holdout.json"
params = json.load(open(params_path, "r"))

A = params["A"]
b = params["b"]
Ea = params["Ea_J_per_mol"]
s_c = params["s_c"]

df_global = pd.DataFrame({
    "Parameter": ["A", "b", "Ea"],
    "Value": [A, b, Ea],
    "Units": ["-", "-", "J/mol"]
})

with open("outputs/tables/table_fitted_global.tex", "w") as f:
    f.write(df_global.to_latex(index=False, float_format="%.6g"))

print("[OK] Wrote outputs/tables/table_fitted_global.tex")

# 4) Severity multipliers: show top 5 and bottom 5
df_sc = pd.DataFrame({
    "condition_id": [int(k) for k in s_c.keys()],
    "s_c": [float(v) for v in s_c.values()]
}).sort_values("s_c")

df_sc_small = pd.concat([df_sc.head(5), df_sc.tail(5)], axis=0)

with open("outputs/tables/table_sc_extremes.tex", "w") as f:
    f.write(df_sc_small.to_latex(index=False, float_format="%.4f"))

print("[OK] Wrote outputs/tables/table_sc_extremes.tex")
