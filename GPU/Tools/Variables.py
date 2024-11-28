class GlobalVariables:
    health_damage_directory = r'E:\uni\Python\OptimizationAlgor\csv_dir\Health Damage Data'
    results_directory = r'E:\uni\Python\OptimizationAlgor\csv_dir\results'

    data_center = ['aws', 'azure', 'gcp']
    consumption_file = ['svm', 'autoencoder_corrected', 'hf_sca_consumption_5', 'isolation_forrest']

    time_window = [6 * 60, 12 * 60, 18 * 60, 24 * 60]  # espresso in minuti
    time_window_percentage = [25, 50, 75, 100]
    definitive_time_window = time_window + time_window_percentage
    checking_time_window = [15, 30, 45, 60, 120]  # espresso in minuti

    kwH_per_GB = 0.028  # Kwh/Gb
    dataset_dim = 0.1  # Gb
    kwH_for_dataset = kwH_per_GB * dataset_dim
