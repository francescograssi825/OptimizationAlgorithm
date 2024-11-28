import itertools

import pandas as pd
import cupy as cp
from concurrent.futures import ProcessPoolExecutor


def get_csv(path, path2=None, header=None):
    df1 = pd.read_csv(path, header=header)
    if path2 is not None:
        df2 = pd.read_csv(path2, header=header)
        df1 = concat_df(df1, df2)
    return df1


def csv_to_array(df, column_index):
    array = df.iloc[:, column_index].values
    gpu_array = cp.array(array, dtype=cp.float64)
    return gpu_array


def list_to_array(list1):
    return cp.asarray(list1, dtype=cp.float64)


def list_to_array_int(list1):
    return cp.asarray(list1, dtype=cp.int32)


def get_dot(l1, l2):
    return cp.dot(l1, l2)


def concat_df(df1, df2):
    return pd.concat([df1.iloc[:-1], df2], ignore_index=True)


def array_result(n):
    return cp.zeros(n, dtype=cp.float64)


def total_sum(r):
    return cp.sum(r)


def invert_convertion(r):
    return cp.asnumpy(r)


def tmp_min_arry(array_size):
    return cp.full(array_size.shape, cp.inf, dtype=cp.float64)


def get_inf():
    return cp.inf


def get_min(a, b):
    return cp.minimum(a, b)


def get_tmp():
    return cp.float64(0)


def get_empty_array(n):
    return cp.zeros(n, dtype=cp.float64)


def concatenate_array(arr1, arr2):
    return cp.concatenate([arr1, arr2], axis=0)


def add_zero_to_array(arr, n):
    return concatenate_array(arr, cp.zeros(int(n), dtype=cp.float64))


def calculate_total_sum(values, lengths_perm, start_index, end_index):
    # Considera solo il sottoarray di values da start_index a end_index
    sub_values = values[start_index:end_index + 1]
    m = len(sub_values)
    n = len(lengths_perm)

    # Trasferisci i dati su GPU
    values_gpu = cp.array(sub_values)

    # Precompute prefix sums
    prefix_sum = cp.zeros(m + 1)
    cp.cumsum(values_gpu, out=prefix_sum[1:])

    # Function to get sum of subarray [start, end]
    def get_sum(start, end):
        return prefix_sum[end + 1] - prefix_sum[start]

    # Initialize dp array with infinity
    dp = cp.full((n, m), cp.inf)

    # Fill dp for the first interval
    for j in range(lengths_perm[0] - 1, m):
        dp[0, j] = get_sum(0, j)

    # Fill dp for other intervals
    for i in range(1, n):
        for j in range(lengths_perm[i] - 1 + i, m):
            for k in range(lengths_perm[i - 1] - 1 + i - 1, j - lengths_perm[i] + 1):
                dp[i, j] = cp.minimum(dp[i, j], dp[i - 1, k] + get_sum(k + 1, j))

    # Get the minimum value in the last row
    return cp.min(dp[n - 1]).get(), lengths_perm


def min_sum_intervals(values, lengths, start_index, end_index):
    # Usa un dizionario per memorizzare i risultati già calcolati

    # Usa una lista di permutazioni uniche
    unique_perms = set(itertools.permutations(lengths))

    min_sum = float('inf')
    best_perm = None

    # Usa ProcessPoolExecutor per parallelizzare le valutazioni delle permutazioni
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(calculate_total_sum, values, perm, start_index, end_index): perm
            for perm in unique_perms
        }
        for future in futures:
            total_sum1, perm = future.result()
            if total_sum1 < min_sum:
                min_sum = total_sum1
                best_perm = perm

    return min_sum, best_perm


def pause_and_resume(lista_valori, lista_intervalli, start_idx, end_idx):
    indici_occupati = set()  # Per tenere traccia degli indici già utilizzati
    indici_assoluti = set()

    # Restringi la lista ai valori tra gli indici di start e end
    sublista_valori = lista_valori[start_idx:end_idx + 1]

    for intervallo in lista_intervalli:
        # Trova il miglior intervallo che non usa indici occupati
        risultato = trova_intervallo_minimo_con_indici(sublista_valori, intervallo, indici_occupati)

        if risultato[1] is not None:  # Se è stato trovato un intervallo valido
            somma_minima, migliori_indici = risultato

            # Converti gli indici relativi in indici assoluti rispetto alla lista originale
            indici_assoluti.update(range((migliori_indici[0] + start_idx), (migliori_indici[1] + 1 + start_idx)))

            # Aggiungi gli indici trovati agli indici occupati
            indici_occupati.update(range(migliori_indici[0], migliori_indici[1] + 1))
    if len(indici_assoluti) != sum(lista_intervalli):
        indici_assoluti = list(range(start_idx, start_idx + sum(lista_intervalli)))

    return sorted(indici_assoluti), indici_occupati


def trova_intervallo_minimo_con_indici(lista_valori, n, indici_occupati):
    somma_minima = float('inf')
    migliori_indici = None

    for i in range(len(lista_valori) - n + 1):
        if any(index in indici_occupati for index in range(i, i + n)):
            continue

        somma_corrente = cp.sum(lista_valori[i:i + n])

        if somma_corrente < somma_minima:
            somma_minima = somma_corrente
            migliori_indici = (i, i + n - 1)

    return somma_minima, migliori_indici
