from src.full_pipeline import run_full_pipeline

if __name__ == "__main__":

    run_full_pipeline("NGC_6809", 
                      show_optical_plots=True, 
                      show_xray_plot=True, 
                      show_crossmatch_DaraFrame=True, 
                      show_probability_DataFrame=True, 
                      search_secure_counterparts=False)
