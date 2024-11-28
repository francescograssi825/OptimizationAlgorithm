import gc

from matplotlib import pyplot as plt

from Utils import *
import pandas as pd
import seaborn as sns


def consumption_value(c_name='svm', time='2021-01-01T08:00:00+00:00', dc='aws', region='BPA(Boardman, Oregon)',
                      lower_index=0, upper_index=None, consumption_lower_index=0, consumption_upper_index=None,
                      df_cons=None):
    # Define file paths for emissions and consumption data
    file_path_emission1 = f'csv_dir/Health Damage Data/{dc.upper()}/{region}/csv/response_2021_{str(extract_month(time)).zfill(2)}.csv'
    if df_cons is None:
        file_path_consumption = f'csv_dir/{c_name}_incremental.csv'
        df2 = pd.read_csv(file_path_consumption, header=0)
    else:
        df2 = df_cons

    # Calculate starting index for the first DataFrame
    #start_index = get_index_by_value(file_path_emission1, 0, time) + lower_index
    start_index = lower_index

    # Load the first DataFrame (emissions), skipping the header row
    df1 = pd.read_csv(file_path_emission1, header=0)  # Assume first row is header

    # Verify that the second DataFrame has enough rows
    if len(df2) == 0:
        raise ValueError("Il secondo file CSV è vuoto.")

    # Verify that the first DataFrame has enough rows for starting from the lower_index
    if start_index >= len(df1):
        raise ValueError("L'indice di partenza è maggiore del numero di righe del primo DataFrame.")

    # Filter the first DataFrame starting from the lower_index
    df1_filtered = df1.iloc[start_index:].reset_index(drop=True)

    # Apply the upper_index if specified
    if upper_index is not None:
        # Ensure that the upper_index is relative to the start of the first DataFrame
        if upper_index < lower_index:
            raise ValueError("upper_index deve essere maggiore o uguale all'indice inferiore.")
        upper_index -= lower_index
        df1_filtered = df1_filtered.iloc[:upper_index - lower_index]

    # Calculate starting index for the second DataFrame
    if consumption_upper_index is not None:
        df2 = df2.iloc[consumption_lower_index:consumption_upper_index].reset_index(drop=True)
    else:
        df2 = df2.iloc[consumption_lower_index:].reset_index(drop=True)

    # Initialize the starting point for the second DataFrame
    second_df_index = 0

    total_sum = 0

    while len(df2) > second_df_index and len(df1_filtered) > 0:
        # Check if there are enough rows in the filtered first DataFrame
        if len(df1_filtered) < len(df2) - second_df_index:
            # If not enough rows in the first DataFrame, use the third DataFrame
            if len(df1_filtered) > 0:
                # Align the first DataFrame with the second DataFrame
                df1_aligned = df1_filtered.reset_index(drop=True)
                col1_df1 = df1_aligned.iloc[:, 1]
                col2_df2 = df2.iloc[second_df_index:second_df_index + len(df1_filtered)].reset_index(drop=True).iloc[:,
                           3]

                total_sum += sum(float(col1_df1[i]) * float(col2_df2[i]) for i in range(len(col1_df1)))

                # Update the index of the second DataFrame
                second_df_index += len(df1_filtered)
                df1_filtered = pd.DataFrame()  # Clear the filtered DataFrame

            # Load the third DataFrame if necessary
            file_path_emission2 = f'csv_dir/Health Damage Data/{dc.upper()}/{region}/csv/response_2021_{str(extract_month(time) + 1).zfill(2)}.csv'
            df3 = pd.read_csv(file_path_emission2, header=0)  # Third DataFrame

            # Verify that the third DataFrame has rows
            if len(df3) == 0:
                break

            # Calculate remaining rows to process
            remaining_rows = (upper_index - lower_index) - (second_df_index + start_index - lower_index)

            # Limit the third DataFrame to the required rows
            df1_filtered = df3.iloc[:remaining_rows].reset_index(drop=True)
        else:
            # Continue with the first DataFrame
            df1_aligned = df1_filtered.iloc[:len(df2) - second_df_index].reset_index(drop=True)
            col1_df1 = df1_aligned.iloc[:, 1]
            col2_df2 = df2.iloc[second_df_index:second_df_index + len(df1_aligned)].reset_index(drop=True).iloc[:, 3]

            total_sum += sum(float(col1_df1[i]) * float(col2_df2[i]) for i in range(len(col1_df1)))

            # Update the index of the second DataFrame
            second_df_index += len(df1_aligned)
            df1_filtered = df1_filtered[len(df1_aligned):]

    return total_sum


def interval_value_df(df1, df2, df_consumption, time, total_time=None):
    if total_time is None:
        total_time = 23 * 60
    if df1 is None or df_consumption is None:
        print("Data frame null")
        return None
    index_1 = get_index_by_value_df(df1, 0, time)
    dimension_consumption = count_rows_csv_df(df_consumption)
    total_sum = float('inf')
    best_time = time

    if df2 is None:

        max_start_index = get_index_by_value_df(df1, 0, add_minutes_to_date(time, total_time).replace('2022', '2021'))

        while index_1 <= max_start_index:
            tmp = 0
            for i in range(dimension_consumption):
                tmp += get_value_by_index_df(df1, index_1 + i, 1) * get_value_by_index_df(df_consumption, i, 3)

            if tmp < total_sum:
                total_sum = tmp
                best_time = get_value_by_index_df(df1, index_1, 0)
            index_1 += 1
        return total_sum, best_time

    else:
        max_start_index = get_index_by_value_df(df2, 0, add_minutes_to_date(time, total_time))
        dimension_df1 = count_rows_csv_df(df1)
        index_2 = 0

        while index_1 <= dimension_df1 and index_2 <= max_start_index:
            tmp = 0
            for i in range(dimension_consumption):
                if index_1 < dimension_df1:
                    tmp += get_value_by_index_df(df1, index_1, 1) * get_value_by_index_df(df_consumption, i, 3)

                else:
                    tmp += get_value_by_index_df(df2, index_2, 1) * get_value_by_index_df(df_consumption, i, 3)

            if total_sum is None or tmp < total_sum:
                total_sum = tmp

            if index_1 < dimension_df1:
                index_1 += 1
            else:
                index_2 += 1

        if index_2 == 0:
            best_time = get_value_by_index_df(df1, index_1 - 1, 0)
        else:
            best_time = get_value_by_index_df(df2, index_2 - 1, 0)

        return total_sum, best_time


def interval_value_pause_and_resume_df(df1, df_consumption, time, control_time, total_time=None):
    if total_time is None:
        total_time = 23 * 60
    print(time)
    print('total ntime ', total_time)
    print(add_minutes_to_date(time, total_time))
    print(get_index_by_value_df(df1, 0, add_minutes_to_date(time, total_time).replace('2022', '2021')))
    if df1 is None or df_consumption is None:
        print("Data frame null")
        return None
    dimension_consumption = count_rows_csv_df(df_consumption)
    dimension_df1 = count_rows_csv_df(df1)
    interval_value_list = interval_list_value(dimension_consumption, control_time)
    i = 0
    occupied_position = []
    sum_total = 0

    for val in range(0, len(interval_value_list)):
        #print('val', val)
        #print('i ', i)
        tmp = best_interval(df1, df_consumption, interval_value_list[val], occupied_position,
                            get_index_by_value_df(df1, 0,
                                                  add_minutes_to_date(time, total_time).replace('2022',
                                                                                                '2021')) - sum_last_n_elements(
                                interval_value_list, len(interval_value_list) - val),
                            get_index_by_value_df(df1, 0, time), val * interval_value_list[val])
        #print('last n elemnet sum ' , sum_last_n_elements(
        #                       interval_value_list, len(interval_value_list) - (val )))
        sum_total += tmp[0]
        if tmp[1] is not None:
            #print('index start calc ', tmp[1])
            for j in range(0, interval_value_list[val]):
                occupied_position.append(tmp[1] + j)

        i += 1

        if i == len(interval_value_list) - 1:
            index_of_end = get_index_by_value_df(df1, 0, add_minutes_to_date(time, total_time).replace('2022', '2021'))
            #print('index of end ',index_of_end)

            #print("cdfd", sum(interval_value_list[:-1]))
            sum_total += best_interval(df1, df_consumption, interval_value_list[len(interval_value_list) - 1],
                                       occupied_position, index_of_end, max(occupied_position),
                                       sum(interval_value_list[:-1]))[0]
            break

    return sum_total


def best_interval(df1, df_consumption, number_of_elements, occupied_position, last_start_index, first_index=0,
                  start_index_consumption=0):
    print(last_start_index)
    #print('number_of_elements', number_of_elements, )
    #print('occupied_position', occupied_position, )
    #print('last_start_index', last_start_index, )
    min_value = None
    index_start_calc = None
    #print('first_index ', first_index)
    #print('start_index_consumption ', start_index_consumption)

    for i in range(first_index, last_start_index):

        if not (all((i + j) not in occupied_position for j in range(number_of_elements))):
            i += 1

        else:

            tmp = calculate_health_damage(df1, df_consumption, i, last_start_index, start_index_consumption,
                                          number_of_elements)

            if min_value is None or (tmp is not None and tmp < min_value):
                min_value = tmp
                #print('min_value', min_value)
                index_start_calc = i

    # if index_start_calc is not None:
    #   print('index start calc ', index_start_calc)
    #    for i in range(0, number_of_elements):
    #       occupied_position.append(index_start_calc + i)
    #print('end min_value', min_value)
    return min_value, index_start_calc


def calculate_health_damage(df1, df_consumption, start_index, end_index, start_index_consumption=0, stop_index=None):
    total_sum = None
    x = 0
    y = count_rows_csv_df(df_consumption)
    print('start_index', start_index, 'end_index', end_index, ' stop index ', stop_index)
    if end_index - start_index >= stop_index:
        total_sum = 0
        print('start_index', start_index, 'end_index', end_index)
        for i in range(end_index - start_index):

            total_sum += get_value_by_index_df(df1, start_index + i, 1) * get_value_by_index_df(df_consumption,
                                                                                                x + start_index_consumption,
                                                                                                3)
            x += 1
            if x + start_index_consumption == y or x == stop_index:
                break

    return total_sum


def interval_value(file_path1, file_path2, cond, min_index, max_index, time='2021-01-01T08:00:00+00:00', c_name='svm'):
    if not cond:
        return consumption_value(c_name, time.replace('2022', '2021'), get_dc_from_path(file_path1).lower(),
                                 get_region_from_path(file_path1),
                                 min_index, max_index, 0, None)
    else:
        val = consumption_value(c_name, time.replace('2022', '2021'), get_dc_from_path(file_path1).lower(),
                                get_region_from_path(file_path1),
                                min_index, None, 0, min_index - max_index)
        val += consumption_value(c_name, add_minutes_to_date(time, 5 * (min_index - max_index)).replace('2022', '2021'),
                                 get_dc_from_path(file_path2).lower(), get_region_from_path(file_path2), 0, None,
                                 min_index - max_index + 1, None)
        return val


def get_csv_month(month=1, dc='aws', consumption='svm.csv'):
    if month < 1 or month > 12:
        print('Invalid month')
        return None
    consumption_df = pd.read_csv('csv_dir/' + consumption)
    consumption_df['timestamp'] = pd.to_datetime(consumption_df['timestamp'])
    consumption_df['timestamp'] = consumption_df['timestamp'].dt.tz_localize(None)
    # Arrotonda i timestamp dei consumi al più vicino intervallo di 5 minuti
    consumption_df['timestamp_rounded'] = consumption_df['timestamp'].dt.round('5min')
    merged_df = pd.DataFrame()
    for region in get_region(dc):
        month_str = str(month).zfill(2)
        tmp = pd.read_csv(f'csv_dir/Health Damage Data/{dc.upper()}/{region}/csv/response_2021_{month_str}.csv')
        tmp['data__point_time'] = pd.to_datetime(tmp['data__point_time'])

        # Uniforma i fusi orari
        tmp['data__point_time'] = tmp['data__point_time'].dt.tz_convert(None)

        merged = merge_health_damage(consumption_df, tmp, region)
        merged_df[f'health_damage_usd_per_mwh_{region}'] = merged[f'health_damage_usd_per_mwh_{region}']

    if 'total_consumption' not in merged_df.columns:
        merged_df['total_consumption'] = consumption_df[
            'total_consumption']

    merged_df = add_date_column(merged_df, 'timestamp_rounded')

    return merged_df


def get_min_health_damage_value_from_data(dc='aws', time='2021-01-01T00:00:00+00:00'):
    month = extract_month(time.replace('2022', '2021'))
    if month < 1 or month > 12:
        print('Invalid month')
        return None
    best_region = None
    best_value = None
    for region in get_region(dc):
        month_str = str(month).zfill(2)
        print(f'csv_dir/Health Damage Data/{dc.upper()}/{region}/csv/response_2021_{month_str}.csv')
        tmp = pd.read_csv(f'csv_dir/Health Damage Data/{dc.upper()}/{region}/csv/response_2021_{month_str}.csv',
                          usecols=[0, 1])
        riga = tmp[tmp[tmp.columns[0]] == time.replace('2022', '2021')]

        if not riga.empty:
            value = riga[tmp.columns[1]].iloc[0]
            if best_value is None or value < best_value:
                best_value, best_region = value, region

    return best_region, best_value, time


def convert_to_incremental_values(file_name='svm'):
    df = pd.read_csv(f'csv_dir/' + file_name + '.csv')

    # Seleziona solo le colonne numeriche per il calcolo della differenza
    numeric_cols = df.select_dtypes(include='number').columns

    # Converti tutte le colonne selezionate in numeriche, ignorando errori per eventuali dati non convertibili
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Calcola la differenza cumulativa solo sulle colonne numeriche
    df_incremental = df[numeric_cols].diff()

    # Sostituisci i NaN nella prima riga con i valori originali
    df_incremental.iloc[0] = df[numeric_cols].iloc[0]

    # Aggiungi colonne non numeriche al DataFrame risultante, se necessario
    non_numeric_cols = df.select_dtypes(exclude='number').columns
    df_incremental[non_numeric_cols] = df[non_numeric_cols]

    # Salva il risultato in un nuovo file CSV
    df_incremental.to_csv(f'csv_dir/' + file_name + '_incremental.csv', index=False)
    return f'csv_dir/' + file_name + '_incremental.csv'


def save_to_csv(df, filename):
    df.to_csv(filename, index=False)  # `index=False` evita di salvare l'indice come colonna


# Funzione per unire i dati di health damage con i consumi
def merge_health_damage(consumption_df, region_df, region_name):
    merged_df = pd.merge_asof(consumption_df.sort_values('timestamp_rounded'),
                              region_df[['data__point_time', 'data__value']],
                              left_on='timestamp_rounded',
                              right_on='data__point_time',
                              direction='backward')
    merged_df = merged_df.rename(columns={'data__value': f'health_damage_usd_per_mwh_{region_name}'})
    if 'timestamp_rounded' not in merged_df.columns:
        print("La colonna 'timestamp_rounded' non è presente nel DataFrame risultante dalla fusione.")
    return merged_df


def add_date_column(df, timestamp_col):
    if timestamp_col not in df.columns:
        print(f"Colonna '{timestamp_col}' non trovata nel DataFrame")
        return df
    df['date'] = df[timestamp_col].dt.date
    return df


def get_index_by_value(file_path, column_index, value_to_find):
    # Read csv file
    df = pd.read_csv(file_path)

    # control column index
    if column_index < 0 or column_index >= len(df.columns):
        raise IndexError(f"Indice della colonna {column_index} è fuori intervallo.")

    # get name of comlumn
    column_name = df.columns[column_index]

    # find the row with the value
    row = df[df[column_name] == value_to_find]

    del df
    gc.collect()

    if not row.empty:
        # return index of row
        return row.index[0]
    else:
        raise ValueError(f"Il valore '{value_to_find}' non è stato trovato nella colonna con indice '{column_index}'.")


def get_index_by_value_df(df, column_index, value_to_find):
    # control column index
    if column_index < 0 or column_index >= len(df.columns):
        raise IndexError(f"Indice della colonna {column_index} è fuori intervallo.")
    # get name of comlumn
    column_name = df.columns[column_index]
    # find the row with the value
    row = df[df[column_name] == value_to_find]

    if not row.empty:
        # return index of row
        return row.index[0]
    else:
        raise ValueError(f"Il valore '{value_to_find}' non è stato trovato nella colonna con indice '{column_index}'.")


def count_rows_csv(file_path):
    try:
        # Read cvs file
        df = pd.read_csv(file_path)

        # count the number of rows
        num_rows = len(df)

        del df
        gc.collect()

        return num_rows
    except Exception as e:
        print(f"Error: {e}")
        return None


def count_rows_csv_df(df):
    try:

        # count the number of rows
        num_rows = len(df)

        return num_rows
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_value_by_index(file_path, row_index, column_index):
    # read csv file
    df = pd.read_csv(file_path)

    # control row index
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Indice di riga {row_index} è fuori intervallo. Il DataFrame ha {len(df)} righe.")

    # control column index
    if column_index < 0 or column_index >= len(df.columns):
        raise IndexError(
            f"Indice di colonna {column_index} è fuori intervallo. Il DataFrame ha {len(df.columns)} colonne.")

    # get value
    value = df.iloc[row_index, column_index]

    del df
    gc.collect()

    return value


def get_value_by_index_df(df, row_index, column_index):
    # control row index
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Indice di riga {row_index} è fuori intervallo. Il DataFrame ha {len(df)} righe.")

    # control column index
    if column_index < 0 or column_index >= len(df.columns):
        raise IndexError(
            f"Indice di colonna {column_index} è fuori intervallo. Il DataFrame ha {len(df.columns)} colonne.")

    # get value
    value = df.iloc[row_index, column_index]

    return value


def get_value_by_data(file_path, column_index, data_to_find):
    # read csv file

    df = pd.read_csv(file_path)

    ind = get_index_by_value_df(df, column_index, data_to_find)
    val = get_value_by_index_df(df, ind, 1)

    del df
    gc.collect()

    return val


def get_csv(path, header=None):
    if header is None:
        return pd.read_csv(path)
    else:
        return pd.read_csv(path, header=header)


def close_df(df):
    del df
    gc.collect()


def concat_df(df1, df2):
    return pd.concat([df1.iloc[:-1], df2], ignore_index=True)


def save_results(time, data, con_name, dc, val=None, path=None):
    df = pd.DataFrame(data)
    if path is None:
        if val is None:
            folder_path = f'csv_dir/results/{con_name}/{dc}/{str(extract_month(time)).zfill(2)}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            csv_filename = os.path.join(folder_path, f'{df.columns[1]}.csv')
        else:
            folder_path = f'csv_dir/results/{con_name}/{dc}/{str(extract_month(time)).zfill(2)}/day'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            csv_filename = os.path.join(folder_path, f'{df.columns[1]}_{extract_day(time)}.csv')
    else:
        folder_path = path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        csv_filename = os.path.join(folder_path, f'{df.columns[0]}.csv')
    print(csv_filename)

    df.to_csv(csv_filename, index=False)


""""
    png_filename = os.path.join(folder_path, f'health_damage_optimization_results_month_{current_month}.png')

    df.to_csv(csv_filename, index=False)

    plt.figure(figsize=(14, 8))
    sns.lineplot(x='Date', y='value', hue='variable', data=pd.melt(df, ['Date']))
    plt.title(f'Health Damage Over Time with Different Optimization Strategies - Month {current_month}')
    plt.xlabel('Date')
    plt.ylabel('Health Damage')
    plt.legend(title='Strategy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(png_filename)
    plt.show()
"""""


def merge_all_csv_in_folder(folder_path, output_filename):
    # Ottieni la lista di tutti i file CSV nella cartella
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Inizializza un DataFrame vuoto per memorizzare i risultati
    combined_df = None

    # Loop attraverso tutti i file CSV e uniscili
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Se è il primo file, inizia il DataFrame combinato con questo
        if combined_df is None:
            combined_df = df
        else:
            # Unisci i DataFrame sulla colonna comune
            combined_df = pd.merge(combined_df, df, on='Date')

        #os.remove(file_path)

    # Salva il DataFrame combinato in un nuovo file CSV
    output_path = os.path.join(folder_path, output_filename)
    combined_df.to_csv(output_path, index=False)


def file_name():
    # Carica il file CSV
    file_path = 'csv_dir/hf_sca_consumption_incremental.csv'
    df = pd.read_csv(file_path)

    # Controlla le prime righe del DataFrame per capire la struttura
    print(df.head())

    # Assicurati che la colonna che contiene i timestamp sia in formato datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Imposta il timestamp di partenza, ad esempio il primo timestamp del file
    start_time = df['timestamp'].min()

    # Filtra i dati ogni 5 minuti
    filtered_df = df.set_index('timestamp').resample('5T').first().reset_index()

    # Mostra il risultato
    print(filtered_df.head())

    # Se desideri salvare il risultato in un nuovo file CSV
    filtered_df.to_csv('csv_dir/hf_sca_consumption_5_incremental.csv', index=False)


def corr_file():
    file_path = 'csv_dir/autoencoder.csv'
    df = pd.read_csv(file_path)

    # Identifica la colonna con i consumi
    consumption_column = 'total_consumption'  # Sostituisci con il nome esatto della colonna nel tuo file

    # Trova la riga con la discrepanza
    for i in range(1, len(df)):
        if df.loc[i, consumption_column] < df.loc[i - 1, consumption_column]:
            discrepancy_index = i
            break

    # Correggi i valori successivi sommando la differenza
    correction_value = df.loc[discrepancy_index - 1, consumption_column]

    for i in range(discrepancy_index, len(df)):
        df.loc[i, consumption_column] += correction_value

    # Salva il file corretto
    corrected_file_path = 'csv_dir/autoencoder_corrected.csv'
    df.to_csv(corrected_file_path, index=False)


def move_timestamp_to_last(input_file_, output_file_):
    # Read the CSV file
    df = pd.read_csv(input_file_)

    # Move the 'timestamp' column to the last position
    columns = [col for col in df.columns if col != 'timestamp'] + ['timestamp']
    df = df[columns]

    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_file_, index=False)


def find_best_intervals(df, arr, interval_lengths, df_cons):
    # Questa lista conterrà gli intervalli migliori per ciascuna lunghezza
    best_intervals = []
    occupied_positions = set()

    start_multiplier = 0
    for idx, interval_length in enumerate(interval_lengths):
        interval_sums = []

        # Scorri attraverso l'array e calcola la somma pesata per ogni intervallo
        for i in range(len(arr) - interval_length + 1):
            if all(arr[i + j] not in occupied_positions for j in range(interval_length)):
                weighted_sum = 0
                for j in range(interval_length):
                    multiplier = start_multiplier + j
                    weighted_sum += get_value_by_index_df(df, i + j, 1) * get_value_by_index_df(df_cons, multiplier, 3)
                interval_sums.append((i, weighted_sum))

        # Trova l'intervallo con la somma pesata più bassa
        if interval_sums:
            best_start_index, best_sum = min(interval_sums, key=lambda x: x[1])
            best_intervals.append((best_start_index, best_sum))

            # Aggiungi l'intervallo migliore agli occupati
            for j in range(interval_length):
                occupied_positions.add(arr[best_start_index + j])

        # Aggiorna il moltiplicatore iniziale per il prossimo intervallo
        start_multiplier += interval_length
    return best_intervals


def find_best_intervals_array(df_array, arr, interval_lengths, df_cons_array):
    # Questa lista conterrà gli intervalli migliori per ciascuna lunghezza
    best_intervals = []
    occupied_positions = set()

    start_multiplier = 0
    for idx, interval_length in enumerate(interval_lengths):
        interval_sums = []

        # Scorri attraverso l'array e calcola la somma pesata per ogni intervallo
        for i in range(len(arr) - interval_length + 1):
            if all(arr[i + j] not in occupied_positions for j in range(interval_length)):
                weighted_sum = 0
                for j in range(interval_length):
                    multiplier = start_multiplier + j
                    weighted_sum += df_array[i + j, 1] * df_cons_array[multiplier, 3]
                interval_sums.append((i, weighted_sum))

        # Trova l'intervallo con la somma pesata più bassa
        if interval_sums:
            best_start_index, best_sum = min(interval_sums, key=lambda x: x[1])
            best_intervals.append((best_start_index, best_sum))

            # Aggiungi l'intervallo migliore agli occupati
            for j in range(interval_length):
                occupied_positions.add(arr[best_start_index + j])

        # Aggiorna il moltiplicatore iniziale per il prossimo intervallo
        start_multiplier += interval_length
    return best_intervals

##########
def old_interval_value_pause_and_resume_df(df1, df_consumption, time, control_time, extra_time):
    if extra_time is None:
        extra_time = 23 * 60

    if df1 is None or df_consumption is None:
        print("Data frame null")
        return None
    dimension_consumption = count_rows_csv_df(df_consumption)
    dimension_df1 = count_rows_csv_df(df1)
    interval_value_list = interval_list_value(dimension_consumption, control_time)
    i = 0
    occupied_position = []
    sum_total = 0

    for val in interval_value_list:
        sum_total += old_best_interval(df1, df_consumption, val, occupied_position,
                                   dimension_df1 - sum_last_n_elements(interval_value_list, len(interval_value_list) - (
                                           i + 1)) - get_index_by_value_df(df1, 0,
                                                                           next_multiple_of_5(add_minutes_to_date(time, extra_time))),
                                   get_index_by_value_df(df1, 0, time), i * val)

        i += 1

        if i == len(interval_value_list) - 1:
            break

    sum_total += old_best_interval(df1, df_consumption, interval_value_list[len(interval_value_list) - 1],
                               occupied_position, dimension_df1 - sum_last_n_elements(interval_value_list,
                                                                                      len(interval_value_list) - 1) - get_index_by_value_df(
            df1, 0, next_multiple_of_5(add_minutes_to_date(time, extra_time))), max(occupied_position), sum(interval_value_list[:-1]))

    return sum_total


def old_best_interval(df1, df_consumption, number_of_elements, occupied_position, last_start_index, first_index=0,
                  start_index_consumption=0):
    min_value = float('inf')
    index_start_calc = None
    for i in range(first_index, count_rows_csv_df(df1) - last_start_index):
        if not (all((i + j) not in occupied_position for j in range(number_of_elements))):
            i += number_of_elements
        else:
            tmp = old_calculate_health_damage(df1, df_consumption, i, last_start_index, start_index_consumption,
                                          number_of_elements)
            if tmp < min_value:
                min_value = tmp
                index_start_calc = i
    if index_start_calc is not None:
        for i in range(0, number_of_elements):
            occupied_position.append(index_start_calc + i)

    return min_value if min_value < float('inf') else 0


def old_calculate_health_damage(df1, df_consumption, start_index, end_index, start_index_consumption=0, stop_index=None):
    total_sum = 0
    x = 0
    y = count_rows_csv_df(df_consumption)
    for i in range(end_index - start_index):

        total_sum += get_value_by_index_df(df1, start_index + i, 1) * get_value_by_index_df(df_consumption,
                                                                                            x + start_index_consumption,
                                                                                            3)
        x += 1
        if x + start_index_consumption == y or x == stop_index:
            break

    return total_sum


def pause_and_resume(lista_valori, lista_intervalli, start_idx, end_idx):
    indici_occupati = set()  # Per tenere traccia degli indici già utilizzati
    indici_assoluti = set()
    # Restringi la lista ai valori tra gli indici di start e end
    sublista_valori = lista_valori[start_idx:end_idx + 1]

    i = 0
    for intervallo in lista_intervalli:
        print('Intervallo:', intervallo)

        # Trova il miglior intervallo che non usa indici occupati
        risultato = trova_intervallo_minimo_con_indici(sublista_valori, intervallo, indici_occupati)

        if risultato[1] is not None:  # Se è stato trovato un intervallo valido
            somma_minima, migliori_indici = risultato
            print('Intervallo migliore:', somma_minima)
            print('Indici relativi nella sublista:', migliori_indici)

            # Converti gli indici relativi in indici assoluti rispetto alla lista originale
            indici_assoluti.update(range((migliori_indici[0] + start_idx), (migliori_indici[1] + 1 + start_idx)))
            print('Indici assoluti nella lista originale:', indici_assoluti)

            # Aggiungi gli indici trovati agli indici occupati
            indici_occupati.update(range(migliori_indici[0], migliori_indici[1] + 1))

        print('Indici occupati nella sublista:', indici_occupati)
        print('Iterazione:', i)
        i += 1
    return indici_assoluti, indici_occupati


def trova_intervallo_minimo_con_indici(lista_valori, n, indici_occupati):
    # Inizializza la somma minima con un valore molto alto
    somma_minima = float('inf')
    migliori_indici = None

    # Scorri l'array per trovare l'intervallo con la somma minima che non utilizza indici occupati
    for i in range(len(lista_valori) - n + 1):
        # Verifica se gli indici nell'intervallo sono liberi
        if any(index in indici_occupati for index in range(i, i + n)):
            continue  # Salta questo intervallo se uno degli indici è occupato

        # Calcola la somma del sottointervallo corrente
        somma_corrente = sum(lista_valori[i:i + n])

        # Se trovi una somma più piccola, aggiorna la somma minima e gli indici
        if somma_corrente < somma_minima:
            somma_minima = somma_corrente
            migliori_indici = (i, i + n - 1)

    return somma_minima, migliori_indici