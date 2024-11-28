import concurrent

from ElaborateCsv import count_rows_csv_df, get_index_by_value_df, get_value_by_index_df
from GPU.Tools import Variables as v
from GPU.Tools.csvUtils import *
from Utils import get_region, interval_list_value
from concurrent.futures import ThreadPoolExecutor, as_completed


def compare_dict(dict1, dict2):
    list1 = list(dict1.keys())

    for key in list1:

        if dict1[key]['product'] > dict2[key]['product']:
            dict1[key]['product'] = dict2[key]['product']
            dict1[key]['interval'] = dict2[key]['interval']
            dict1[key]['change'] = dict2[key]['change']
    return dict1


def process_region(region, time_window_list, start_index, cons_len, df, df_cons):
    result_dict = {}

    for i in time_window_list:
        tmp_dict = {}
        for x in v.GlobalVariables.checking_time_window:
            tmp_dict2 = {}
            interval_value_list = interval_list_value(cons_len, x)
            array_values = csv_to_array(df[region], 1)
            min_sum, best_perm = min_sum_intervals(array_values, interval_value_list, start_index, i + start_index)
            val = total_sum(csv_to_array(df_cons, 3) * (min_sum / cons_len))
            tmp_dict2[x] = val
            print('valore x ', x)
        tmp_dict[i] = tmp_dict
        result_dict[region] = tmp_dict
    return result_dict


class Optimization:
    def __init__(self, dc='aws', con_name='svm', df_cons=None, month=1):
        self.df_cons = df_cons
        self.len_cons = count_rows_csv_df(df_cons)
        self.dc = dc
        self.con_name = con_name
        self.month = str(month).zfill(2)
        self.df = {}
        for r in get_region(self.dc, 1):
            tmp_path = rf'{v.GlobalVariables.health_damage_directory}\{self.dc.upper()}\{r}\csv\response_2021_{str(self.month).zfill(2)}.csv'
            if month == 12:
                n = 1
            else:
                n = month + 1
            tmp_path_2 = rf'{v.GlobalVariables.health_damage_directory}\{self.dc.upper()}\{r}\csv\response_2021_{str(n).zfill(2)}.csv'
            self.df[r] = get_csv(tmp_path, tmp_path_2, 0)

    def impact_no_optimization(self, time):
        print('righe df ', count_rows_csv_df(self.df_cons))
        value = 0
        if self.dc == 'aws':
            value = 2
        elif self.dc == 'azure':
            value = 6
        elif self.dc == 'gcp':
            value = 5
        region = get_region(self.dc, 1)[value]
        print('Region: ', region)
        start_index = get_index_by_value_df(self.df[region], 0, time)
        ar1 = csv_to_array(self.df[region], 1)
        ar2 = csv_to_array(self.df_cons, 3)

        n = ar1.shape[0]
        m = ar2.shape[0]

        results = array_result(n)

        i = 0
        while i < m:
            results[i] = ar1[i + start_index] * ar2[i]
            i += 1

        return invert_convertion(total_sum(results))

    def simulate_flexible_start(self, time):
        # Lista delle regioni
        # regions = get_region(self.dc, 1)
        value = 0
        if self.dc == 'aws':
            value = 2
        elif self.dc == 'azure':
            value = 6
        elif self.dc == 'gcp':
            value = 5
        re = get_region(self.dc, 1)[value]
        regions = [re]

        # Definizione della funzione che esegue il calcolo per una singola regione
        def process_region(region):
            time_window_list = [(x / 5 + self.len_cons) for x in v.GlobalVariables.time_window]
            time_window_list_percentage = [int(self.len_cons * (1.00 + x / 100)) for x in
                                           v.GlobalVariables.time_window_percentage]
            time_window_list = time_window_list + time_window_list_percentage
            start_index = get_index_by_value_df(self.df[region], 0, time)
            ar1 = csv_to_array(self.df[region], 1)
            ar2 = csv_to_array(self.df_cons, 3)
            ar3 = list_to_array(time_window_list)

            window_size = ar2.shape[0]

            # Array per memorizzare i minimi per ciascun end_index
            min_values = cp.full(ar3.shape, cp.inf, dtype=cp.float64)

            def compute_min_for_end_index(end_index):
                # Assicurati che end_index sia un valore scalare
                end_index = int(end_index)
                # Array per memorizzare i minimi temporanei
                temp_min = cp.inf
                for i in range(start_index, min(start_index + end_index - window_size + 2, ar1.shape[0])):
                    window = ar1[i:i + window_size]
                    product_sum = total_sum(window * ar2)
                    temp_min = get_min(temp_min, product_sum)

                return temp_min

            # Calcola il minimo per ogni indice di fine e memorizza i risultati
            for idx in range(len(ar3)):
                end_index = ar3[idx]
                min_values[idx] = compute_min_for_end_index(end_index)

            # Trasferimento dei risultati minimi dalla GPU alla CPU
            min_values_cpu = invert_convertion(min_values)
            return region, min_values_cpu  # Ritorna la regione e i minimi per ciascun end_index

        # Esecuzione parallela per tutte le regioni
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_region, regions))

        # Organizza i risultati in un dizionario dove le chiavi sono le regioni e i valori sono array dei minimi
        result_dict = {region: min_values for region, min_values in results}

        return result_dict

    def simulate_static_start_follow_the_sun(self, time):
        # Trovo la regione migliore da cui iniziare
        val = float('inf')
        reg = str('')
        df_arrays_gpu = {}
        index = 0
        for r in get_region(self.dc, 1):
            index = get_index_by_value_df(self.df[r], 0, time)
            tmp = get_value_by_index_df(self.df[r], index, 1)
            df_arrays_gpu[r] = csv_to_array(self.df[r], 1)
            if tmp < val:
                val = tmp
                reg = r

        change = [reg]
        cons_array = csv_to_array(self.df_cons, 3)
        dimension_cons_array = cons_array.shape[0]
        intervals = [int(x / 5) for x in v.GlobalVariables.checking_time_window]
        results = {}  # Struttura per memorizzare i risultati per ogni intervallo

        # Funzione per processare ogni intervallo
        def process_interval(interval, initial_i, initial_reg, cons_array, dimension_cons_array):
            interval_results = []  # Lista per memorizzare i risultati per questo intervallo
            local_change = [initial_reg]
            local_reg = initial_reg
            local_i = initial_i
            local_product = cp.zeros(dimension_cons_array, dtype=cp.float64)  # Inizializza il prodotto

            while local_i < dimension_cons_array:
                try:
                    # Creazione di indici e conversione a tipo intero
                    indices = list_to_array_int(list(range(int(local_i + index), int(local_i + interval + index))))
                    indices_cons = list_to_array_int(list(range(int(local_i), int(local_i + interval))))

                    # Controllo sugli indici
                    if max(indices) >= df_arrays_gpu[local_reg].shape[0]:
                        raise ValueError(f"Indice {max(indices)} fuori dai limiti per df_arrays_gpu[{local_reg}]")
                    if max(indices_cons) >= dimension_cons_array:
                        # Aggiungi zeri al cons_array se gli indici superano la dimensione dell'array
                        cons_array = add_zero_to_array(cons_array, max(indices_cons) - dimension_cons_array + 1)

                    # Accesso agli array usando indici interi
                    s1 = df_arrays_gpu[local_reg][indices]
                    s2 = cons_array[indices_cons]
                    #print(f"Processing interval {interval} with s1 shape: {s1.shape} and s2 shape: {s2.shape}")

                    # Sincronizzazione GPU
                    cp.cuda.Stream.null.synchronize()

                    # Calcolo del prodotto
                    product = s1 * s2
                    local_product[indices_cons] += product  # Accumula il prodotto

                    # Trova la regione con il valore minimo
                    tmp_min = float('inf')
                    r_min = str('')
                    for r in get_region(self.dc, 1):
                        if df_arrays_gpu[r][index + interval + local_i] < tmp_min:
                            tmp_min = df_arrays_gpu[r][index + interval + local_i]
                            r_min = r
                    if r_min != local_reg:
                        product += ((df_arrays_gpu[r_min][index + interval + local_i] + df_arrays_gpu[local_reg][
                            index + interval + local_i]) / 2) * v.GlobalVariables.kwH_for_dataset
                        local_change.append(r_min)
                        local_reg = r_min

                    local_i += interval
                except Exception as e:
                    print(f"Errore durante l'elaborazione dell'intervallo {interval}: {e}")
                    break  # Rompe il ciclo in caso di errore per evitare loop infiniti

            return {
                'interval': interval,
                'change': local_change,
                'product': local_product.get().sum()  # Converte l'array GPU in numpy prima di sommare
            }

        # Elenco dei risultati
        for interval in intervals:
            result = process_interval(interval, 0, reg, cons_array, dimension_cons_array)
            results[interval] = result

        return results

    """"def simulate_flexible_start_follow_the_sun(self, time):
        print(v.GlobalVariables.time_window)

        time_window_list = [int(x / 5 + self.len_cons) for x in v.GlobalVariables.time_window]
        print(time_window_list)
        print(v.GlobalVariables.time_window_percentage)
        time_window_list_percentage = [int(self.len_cons * (1.00 + x / 100)) for x in
                                       v.GlobalVariables.time_window_percentage]
        print(time_window_list_percentage)
        time_window_list = time_window_list + time_window_list_percentage
        print(time_window_list)

        # Ordina la lista dei time window
        new_time_window_list = sorted([int(x - self.len_cons) for x in time_window_list])

        print(new_time_window_list)

        start_index = get_index_by_value_df(self.df[get_region(self.dc, 1)[1]], 0, time)
        cons_len = count_rows_csv_df(self.df_cons)

        # Dizionario per memorizzare i migliori risultati per ogni combinazione di intervallo e valore della lista
        best_results_per_interval = {}

        # Inizializzazione dei risultati
        previous_best_results = {}

        # Funzione interna per eseguire una singola iterazione
        util_region = self.df[get_region(self.dc, 1)[1]]
        i = 0
        j = 0

        last_element = new_time_window_list[-1]

        y = None
        while i < last_element:
            tmp_data = get_value_by_index_df(util_region, i + start_index, 0)
            x = self.simulate_static_start_follow_the_sun(tmp_data)

            if new_time_window_list[j] in best_results_per_interval:  #esegui il confronto per ogni sottochiave
                y = best_results_per_interval[new_time_window_list[j]]
                y = compare_dict(x, y)
                # codice che confronta x e y e restitiosce i valori miglori
            else:
                best_results_per_interval[new_time_window_list[j]] = x
            i += 1
            if i == last_element:
                break
            if i == new_time_window_list[j] and i is not last_element:
                best_results_per_interval[new_time_window_list[j + 1]] = y
                j += 1

        return best_results_per_interval
"""""

    def simulate_flexible_start_follow_the_sun(self, time):

        time_window_list = [int(x / 5 + self.len_cons) for x in v.GlobalVariables.time_window[-2:]]

        time_window_list_percentage = [int(self.len_cons * (1.00 + x / 100)) for x in
                                       v.GlobalVariables.time_window_percentage[-2:]]

        time_window_list = time_window_list + time_window_list_percentage

        # Ordina la lista dei time window
        new_time_window_list = sorted([int(x - self.len_cons) for x in time_window_list])

        start_index = get_index_by_value_df(self.df[get_region(self.dc, 1)[1]], 0, time)
        cons_len = count_rows_csv_df(self.df_cons)

        # Dizionario per memorizzare i migliori risultati per ogni combinazione di intervallo e valore della lista
        best_results_per_interval = {}

        # Inizializzazione dei risultati
        previous_best_results = {}

        # Funzione interna per eseguire una singola iterazione
        util_region = self.df[get_region(self.dc, 1)[1]]
        i = 0
        j = 0

        last_element = new_time_window_list[-1]

        y = None
        while i < last_element:
            tmp_data = get_value_by_index_df(util_region, i + start_index, 0)
            x = self.simulate_static_start_follow_the_sun(tmp_data)

            if new_time_window_list[j] in best_results_per_interval:  # esegui il confronto per ogni sottochiave
                y = best_results_per_interval[new_time_window_list[j]]
                y = compare_dict(x, y)
                # codice che confronta x e y e restitiosce i valori migliori
            else:
                best_results_per_interval[new_time_window_list[j]] = x
            i += 1
            if i == last_element:
                break
            if i == new_time_window_list[j] and i is not last_element:
                best_results_per_interval[new_time_window_list[j + 1]] = y
                j += 1

        return best_results_per_interval

    def simulate_flexible_start_follow_the_sun_2(self, time):

        time_window_list = [int(x / 5 + self.len_cons) for x in v.GlobalVariables.time_window[-2:]]
        time_window_list_percentage = [int(self.len_cons * (1.00 + x / 100)) for x in
                                       v.GlobalVariables.time_window_percentage[-2:]]
        time_window_list = time_window_list + time_window_list_percentage

        # Sort the list of time windows
        new_time_window_list = sorted([int(x - self.len_cons) for x in time_window_list])

        start_index = get_index_by_value_df(self.df[get_region(self.dc, 1)[1]], 0, time)
        cons_len = count_rows_csv_df(self.df_cons)

        # Dictionary to store the best results for each combination of interval and value from the list
        best_results_per_interval = {}

        util_region = self.df[get_region(self.dc, 1)[1]]
        last_element = new_time_window_list[-1]

        # Internal function to handle each iteration
        def process_interval(i, start_index, new_time_window_list, j):
            tmp_data = get_value_by_index_df(util_region, i + start_index, 0)
            x = self.simulate_static_start_follow_the_sun(tmp_data)

            if new_time_window_list[j] in best_results_per_interval:  # compare results
                y = best_results_per_interval[new_time_window_list[j]]
                y = compare_dict(x, y)
                return new_time_window_list[j], y
            else:
                return new_time_window_list[j], x

        # Parallelize with ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = []
            i = 0
            j = 0
            while i < last_element:
                # Submit each task to the thread pool
                futures.append(executor.submit(process_interval, i, start_index, new_time_window_list, j))
                i += 1
                if i == last_element:
                    break
                if i == new_time_window_list[j] and i != last_element:
                    j += 1

            # Gather and store the results
            for future in as_completed(futures):
                try:
                    time_window_key, result = future.result()
                    best_results_per_interval[time_window_key] = result
                except Exception as exc:
                    print(f"Generated an exception: {exc}")

        return best_results_per_interval

    def simulate_flexible_start_follow_the_sun_3(self, time):
        # Convert lists to cupy arrays
        time_window_list = cp.array([int(x / 5 + self.len_cons) for x in v.GlobalVariables.time_window[-2:]])
        time_window_list_percentage = cp.array(
            [int(self.len_cons * (1.00 + x / 100)) for x in v.GlobalVariables.time_window_percentage[-2:]])

        # Concatenate the lists and sort them on GPU
        time_window_list = cp.concatenate((time_window_list, time_window_list_percentage))
        new_time_window_list = cp.sort(time_window_list - self.len_cons)

        start_index = get_index_by_value_df(self.df[get_region(self.dc, 1)[1]], 0, time)
        cons_len = count_rows_csv_df(self.df_cons)

        # Prepare dictionary to store best results
        best_results_per_interval = {}

        util_region = self.df[get_region(self.dc, 1)[1]]
        last_element = new_time_window_list[-1]

        # Function to process each interval
        def process_interval(i, start_index, new_time_window_list, j):
            tmp_data = get_value_by_index_df(util_region, i + start_index, 0)
            x = self.simulate_static_start_follow_the_sun(tmp_data)
            if new_time_window_list[j] in best_results_per_interval:  # Compare results
                y = best_results_per_interval[new_time_window_list[j]]
                y = compare_dict(x, y)
                return new_time_window_list[j], y
            else:
                return new_time_window_list[j], x

        i = 0
        j = 0
        while i < last_element:
            tmp_data = get_value_by_index_df(util_region, i + start_index, 0)
            tmp_data_gpu = cp.asarray(tmp_data)  # Ensure the data is on GPU
            x = self.simulate_static_start_follow_the_sun(tmp_data_gpu)

            if new_time_window_list[j] in best_results_per_interval:
                y = best_results_per_interval[new_time_window_list[j]]
                y_gpu = cp.asarray(y)  # Move existing result to GPU
                y = compare_dict(x, y_gpu)  # Ensure both are GPU arrays during comparison
            else:
                best_results_per_interval[new_time_window_list[j]] = x

            i += 1
            if i == last_element:
                break
            if i == new_time_window_list[j] and i != last_element:
                j += 1

        # Convert results back to CPU if needed
        best_results_per_interval_cpu = {int(k): cp.asnumpy(v) for k, v in best_results_per_interval.items()}

        return best_results_per_interval_cpu

    def simulate_pause_and_resume(self, time):
        #regions = get_region(self.dc, 1)
        value = 0
        if self.dc == 'aws':
            value = 2
        elif self.dc == 'azure':
            value = 6
        elif self.dc == 'gcp':
            value = 5
        re = get_region(self.dc, 1)[value]
        regions = [re]
        # time window list indica il tempo di esecuzione totale
        time_window_list = [int(x / 5 + self.len_cons) for x in v.GlobalVariables.time_window]
        time_window_list_percentage = [int(self.len_cons * (1.00 + x / 100)) for x in
                                       v.GlobalVariables.time_window_percentage]
        print(time_window_list)
        print(time_window_list_percentage)
        # in questo mdo time window list dice quanti indici si sono a disposizione dopo lo start index
        time_window_list = time_window_list + time_window_list_percentage

        time_window_list.sort()

        array_cons = csv_to_array(self.df_cons, 3)

        dict_result = {}

        for region in regions:
            dict_window = {}
            for time_window in time_window_list:
                print(time_window)
                dict_check_time = {}
                for check_time in v.GlobalVariables.checking_time_window:
                    start_index = get_index_by_value_df(self.df[get_region(self.dc, 1)[1]], 0, time)
                    array_df = csv_to_array(self.df[region], 1)
                    interval_list = interval_list_value(self.len_cons, check_time)
                    indices = pause_and_resume(array_df, interval_list, start_index, start_index + time_window)
                    indices = list_to_array_int(list(indices[0]))
                    if len(indices) != len(array_cons):
                        print('start index', start_index)
                        print('end index', start_index + time_window)
                        print('time window ', time_window)
                        print('checking time window', check_time)
                        print('interval list ', interval_list)
                        print('indices ', indices)
                        print('len indices ', len(indices))
                        print('len cons ', len(array_cons))
                    value = get_dot(array_df[indices], array_cons)
                    dict_check_time[check_time] = invert_convertion(value / 1000)
                print(time_window)
                dict_window[time_window] = dict_check_time
                print("keys", dict_window.keys())

            dict_result[region] = dict_window
        return dict_result
