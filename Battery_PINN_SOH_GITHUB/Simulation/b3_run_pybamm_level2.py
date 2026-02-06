import pybamm
import matplotlib.pyplot as plt
from b2_experiment_from_csv import make_experiment
import numpy as np

def voltage_stress_metric(sol):
    V = sol["Terminal voltage [V]"].entries
    # robust scalar proxy: mean absolute deviation from median voltage
    return float(np.mean(np.abs(V - np.median(V))))

def run_one(protocol_csv, temperature="25 oC", n_cycles=3, initial_soc=0.2):
    model = pybamm.lithium_ion.DFN()
    param = pybamm.ParameterValues("Chen2020")

    experiment, cycle_steps = make_experiment(
        protocol_csv,
        n_cycles=n_cycles,
        temperature=temperature
    )

    solver = pybamm.CasadiSolver(mode="safe")
    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        experiment=experiment,
        solver=solver
    )


    sol = sim.solve(initial_soc=initial_soc)

    return sol, cycle_steps


def plot_voltage(sol, title, savepath):
    t = sol["Time [h]"].entries
    V = sol["Terminal voltage [V]"].entries

    plt.figure()
    plt.plot(t, V)
    plt.xlabel("Time [h]")
    plt.ylabel("Terminal voltage [V]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    print("Saved:", savepath)

if __name__ == "__main__":
    # Choose two representative protocols
    mild_csv = "protocol_summaries/DOE-001-050-25DU 08__Channel_25_1.csv"
    harsh_csv = "protocol_summaries/DOE-001-050-60DU 22__Channel_73_1.csv"

    # Run a small number of cycles first just to confirm it runs
    sol_mild, steps_mild = run_one(mild_csv, temperature="25 oC", n_cycles=3)
    sol_harsh, steps_harsh = run_one(harsh_csv, temperature="25 oC", n_cycles=3)
    # --- Temperature comparison (same protocol, different T) ---
    sol_mild_25C, _ = run_one(mild_csv, temperature="25 oC", n_cycles=3, initial_soc=0.2)
    sol_mild_45C, _ = run_one(mild_csv, temperature="45 oC", n_cycles=3, initial_soc=0.2)
    mild_stress = voltage_stress_metric(sol_mild)
    harsh_stress = voltage_stress_metric(sol_harsh)

    print("\nVoltage stress proxy:")
    print("  mild (25DU):", mild_stress)
    print("  harsh (60DU):", harsh_stress)

    plot_voltage(sol_mild_25C, "PyBaMM voltage (25DU, 25C)", "outputs/figures/pybamm_voltage_25DU_25C.png")
    plot_voltage(sol_mild_45C, "PyBaMM voltage (25DU, 45C)", "outputs/figures/pybamm_voltage_25DU_45C.png")

    plot_voltage(sol_mild, "PyBaMM voltage response (25DU, 25C)", "outputs/figures/pybamm_voltage_25DU_25C.png")
    plot_voltage(sol_harsh, "PyBaMM voltage response (60DU, 25C)", "outputs/figures/pybamm_voltage_60DU_25C.png")

    print("\nMild cycle-1 steps:")
    for s in steps_mild: print(" ", s)

    print("\nHarsh cycle-1 steps:")
    for s in steps_harsh: print(" ", s)
