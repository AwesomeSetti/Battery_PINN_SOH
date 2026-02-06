import pandas as pd
import numpy as np

RAW_FILE = "DOE-001-050-10DU 01.xlsx"   # change later to 45DU file too
CHANNEL_SHEET = "Channel_49_1"          # change per sample/channel

def segment_steps(df: pd.DataFrame) -> pd.DataFrame:
    # Make a segment id whenever Step_Index changes
    seg_id = (df["Step_Index"].ne(df["Step_Index"].shift())).cumsum()
    g = df.groupby(seg_id)

    rows = []
    for sid, d in g:
        step = int(d["Step_Index"].iloc[0])
        cyc  = int(d["Cycle_Index"].iloc[0])

        t0 = float(d["Test_Time(s)"].iloc[0])
        t1 = float(d["Test_Time(s)"].iloc[-1])
        dur = max(0.0, t1 - t0)

        I = d["Current(A)"].to_numpy()
        V = d["Voltage(V)"].to_numpy()

        rows.append({
            "seg": int(sid),
            "cycle": cyc,
            "step_index": step,
            "dur_s": dur,
            "I_mean": float(np.mean(I)),
            "I_min": float(np.min(I)),
            "I_max": float(np.max(I)),
            "V_start": float(V[0]),
            "V_end": float(V[-1]),
            "V_min": float(np.min(V)),
            "V_max": float(np.max(V)),
        })
    return pd.DataFrame(rows)

def label_step(r):
    # simple rule-based labeling
    if abs(r["I_mean"]) < 1e-3:
        return "Rest"
    if r["I_mean"] > 0:
        # positive current → charging
        if (r["V_max"] - r["V_min"]) < 0.01:
            return "Hold (CV-ish)"   # voltage ~constant
        return "Charge (CC-ish)"
    else:
        return "Discharge (CC-ish)"

def main():
    df = pd.read_excel(RAW_FILE, sheet_name=CHANNEL_SHEET, engine="openpyxl")
    steps = segment_steps(df)
    steps["type"] = steps.apply(label_step, axis=1)

    # Show cycle 1 only (usually enough to identify protocol)
    cycle1 = steps[steps["cycle"] == steps["cycle"].min()].copy()

    print("\n=== Cycle 1 step summary ===")
    print(cycle1[["seg","step_index","type","dur_s","I_mean","I_min","I_max","V_min","V_max","V_start","V_end"]])

    # Save for report / debugging
    out_csv = f"protocol_summary_{RAW_FILE.replace('.xlsx','')}_{CHANNEL_SHEET}.csv"
    cycle1.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

if __name__ == "__main__":
    main()
