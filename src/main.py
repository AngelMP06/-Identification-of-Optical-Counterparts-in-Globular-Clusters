from src.full_pipeline import run_full_pipeline

if __name__ == "__main__":

    probabilities = run_full_pipeline("NGC_6809", show_optical_plots=False, show_xray_plot=True)

    print(probabilities)

    print("A source is likely to be a counterpart if it has more than 80% on P_counterpart (%)")