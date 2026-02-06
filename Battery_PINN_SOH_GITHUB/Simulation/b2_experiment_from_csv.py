import pandas as pd
import pybamm

def build_steps_from_cycle1_csv(csv_path: str):
    """
    Convert our cycle-1 step summary CSV into PyBaMM experiment step strings.
    Uses current in A (not C-rate) to avoid needing nominal capacity.
    """
    df = pd.read_csv(csv_path)

    steps = []
    for _, r in df.iterrows():
        dur_s = float(r["dur_s"])
        I = float(r["I_mean"])
        vmin = float(r["V_min"])
        vmax = float(r["V_max"])
        step_type = str(r["type"])

        # Skip zero-duration steps (some files have 0s artifacts)
        if dur_s <= 0:
            continue

        if "Rest" in step_type or abs(I) < 1e-3:
            if dur_s < 1:
                # PyBaMM requires step time > 0, so skip zero-length rests
                continue
            steps.append(f"Rest for {int(round(dur_s))} seconds")
            continue


        # Charge step (I > 0)
        if I > 0:
            # If voltage is essentially constant, your labeling called it "Hold (CV-ish)"
            # But your current is large, so we treat it as CC charge until that voltage.
            if (vmax - vmin) < 0.02:
                Vtarget = vmax
                steps.append(f"Charge at {abs(I):.2f} A until {Vtarget:.2f} V")
            else:
                Vtarget = vmax  # charge goes toward upper voltage
                steps.append(f"Charge at {abs(I):.2f} A until {Vtarget:.2f} V")
            continue

        # Discharge step (I < 0)
        if I < 0:
            Vtarget = vmin  # discharge goes toward lower voltage
            steps.append(f"Discharge at {abs(I):.2f} A until {Vtarget:.2f} V")
            continue

    return steps

def make_experiment(csv_path: str, n_cycles: int, temperature: str):
    cycle_steps = build_steps_from_cycle1_csv(csv_path)
    exp = pybamm.Experiment(cycle_steps * n_cycles, temperature=temperature)
    return exp, cycle_steps

if __name__ == "__main__":
    # quick test
    csv = "protocol_summaries/DOE-001-050-25DU 08__Channel_25_1.csv"
    exp, steps = make_experiment(csv_path=csv, n_cycles=2, temperature="25 oC")
    print("Cycle-1 steps:")
    for s in steps:
        print("  -", s)
    print("\nExperiment created OK.")
