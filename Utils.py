import os
from datetime import datetime, timedelta
from dateutil import parser


def extract_month(date_str):
    # Elenco di formati di data da provare
    date_formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # formato con fuso orario
        "%Y-%m-%dT%H:%M:%S",  # formato senza fuso orario
        "%Y-%m-%d %H:%M:%S",  # formato con spazi
        "%Y-%m-%d",  # solo data
        "%d/%m/%Y",  # formato italiano gg/mm/aaaa
        "%m/%d/%Y",  # formato americano mm/gg/aaaa
        "%d-%m-%Y %H:%M:%S",  # data con ore, minuti e secondi
        "%Y/%m/%d %H:%M:%S",  # formato con slash
    ]

    # Analizza la stringa di data e ottieni un oggetto datetime
    date_obj = parse_datetime(date_str, date_formats)

    # Usa strftime per ottenere il nome completo del mese
    month = date_obj.month

    return month


def extract_day(date_str):
    # Elenco di formati di data da provare
    date_formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # formato con fuso orario
        "%Y-%m-%dT%H:%M:%S",  # formato senza fuso orario
        "%Y-%m-%d %H:%M:%S",  # formato con spazi
        "%Y-%m-%d",  # solo data
        "%d/%m/%Y",  # formato italiano gg/mm/aaaa
        "%m/%d/%Y",  # formato americano mm/gg/aaaa
        "%d-%m-%Y %H:%M:%S",  # data con ore, minuti e secondi
        "%Y/%m/%d %H:%M:%S",  # formato con slash
    ]

    # Analizza la stringa di data e ottieni un oggetto datetime
    date_obj = parse_datetime(date_str, date_formats)

    # Usa strftime per ottenere il nome completo del mese
    day = date_obj.day

    return day


# Funzione di supporto per analizzare la data con vari formati
def parse_datetime(dt_str, date_formats):
    for fmt in date_formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date string '{dt_str}' does not match any expected format.")


def get_region(dc='aws', val=None):
    if val is None:
        base_dir = "csv_dir/Health Damage Data/" + dc.upper()
    else:
        base_dir = r"E:\uni\Python\OptimizationAlgor\csv_dir\Health Damage Data/" + dc.upper()

    # Verifica che la directory esista
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"La directory '{base_dir}' non esiste.")

    all_regions = os.listdir(base_dir)

    # for i in range(len(all_regions)):
    #     indice = all_regions[i].find('(')
    #     if indice != -1:
    #         all_regions[i] = all_regions[i][:indice]
    return all_regions


def add_minutes_to_date(date_str, minutes):
    # Definisci i formati di data comuni
    date_formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 con fuso orario
        "%Y-%m-%dT%H:%M:%S",  # ISO 8601 senza fuso orario
        "%Y-%m-%d %H:%M:%S",  # Formato con spazio tra data e ora
        "%d/%m/%Y %H:%M:%S",  # Formato giorno/mese/anno
        "%m/%d/%Y %H:%M:%S"  # Formato mese/giorno/anno
    ]

    original_format = None
    dt = None

    # Prova a parsare la data usando ciascun formato
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            original_format = fmt
            break
        except ValueError:
            continue

    if dt is None:
        raise ValueError(f"Data '{date_str}' non può essere parsata con nessuno dei formati previsti.")

    # Aggiungi i minuti alla data
    new_dt = dt + timedelta(minutes=minutes)

    # Ritorna la nuova data nel formato originale
    return new_dt.strftime(original_format).replace('0000', '00:00')


def subtract_minutes_from_date(date_str, minutes):
    # Definisci i formati di data comuni
    date_formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 con fuso orario
        "%Y-%m-%dT%H:%M:%S",  # ISO 8601 senza fuso orario
        "%Y-%m-%d %H:%M:%S",  # Formato con spazio tra data e ora
        "%d/%m/%Y %H:%M:%S",  # Formato giorno/mese/anno
        "%m/%d/%Y %H:%M:%S"  # Formato mese/giorno/anno
    ]

    original_format = None
    dt = None

    # Prova a parsare la data usando ciascun formato
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            original_format = fmt
            break
        except ValueError:
            continue

    if dt is None:
        raise ValueError(f"Data '{date_str}' non può essere parsata con nessuno dei formati previsti.")

    # Sottrai i minuti dalla data
    new_dt = dt - timedelta(minutes=minutes)

    # Ritorna la nuova data nel formato originale
    return new_dt.strftime(original_format).replace('0000', '00:00')


def compare_dates_month(date1_str, date2_str):
    # Definisci una lista di formati di data che vuoi supportare
    possible_formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 con fuso orario
        "%Y-%m-%dT%H:%M:%S",  # ISO 8601 senza fuso orario
        "%Y-%m-%d %H:%M:%S",  # Formato comune con spazio
        "%Y-%m-%d",  # Solo data
        "%d/%m/%Y",  # Formato europeo giorno/mese/anno
        "%m/%d/%Y",  # Formato americano mese/giorno/anno
        "%d-%m-%Y",  # Formato con trattini
        "%m-%d-%Y",  # Mese-giorno-anno con trattini
    ]

    # Interpreta le date usando i formati possibili
    date1 = parse_datetime(date1_str, possible_formats)
    date2 = parse_datetime(date2_str, possible_formats)

    # Estrai l'anno e il mese per il confronto
    year_month1 = (date1.year, date1.month)
    year_month2 = (date2.year, date2.month)

    # Confronta i tuple di anno e mese
    if year_month1 > year_month2:
        return 1  # "Date 1 è maggiore (mese/anno successivo)"
    elif year_month1 < year_month2:
        return 2  # "Date 2 è maggiore (mese/anno successivo)"
    else:
        return 0  # "Entrambe le date sono nello stesso mese e anno"


def compare_dates(date_str1, date_str2):  # return true if date 1 > date 2
    # Define common date formats
    date_formats = [
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 with timezone
        "%Y-%m-%dT%H:%M:%S",  # ISO 8601 without timezone
        "%Y-%m-%d %H:%M:%S",  # Format with space between date and time
        "%d/%m/%Y %H:%M:%S",  # Day/Month/Year format
        "%m/%d/%Y %H:%M:%S"  # Month/Day/Year format
    ]

    def parse_date(date_str):
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Date '{date_str}' cannot be parsed with any of the supported formats.")

    # Parse both dates
    date1 = parse_date(date_str1)
    date2 = parse_date(date_str2)

    # Compare the two dates and return the larger one in the same format as the input
    if date1 > date2:
        return True
    else:
        return False


def get_next_day(date_str, hour=0, minute=0, second=0, microsecond=0):
    # Analizza la stringa di data in un oggetto datetime
    date = parser.parse(date_str)

    # Calcola la mezzanotte del giorno successivo
    next_midnight = (date + timedelta(days=1)).replace(hour=hour, minute=minute, second=second, microsecond=microsecond,
                                                       year=date.year)
    # Ritorna la data formattata come stringa
    return next_midnight.isoformat()


def get_dc_from_path(path):
    path_parts = path.split('/')
    # Trova l'indice della cartella 'Health Damage Data'
    try:
        health_damage_index = path_parts.index('Health Damage Data')
        # Restituisci l'elemento successivo dopo 'Health Damage Data'
        if health_damage_index + 1 < len(path_parts):
            return path_parts[health_damage_index + 1]
        else:
            raise ValueError("Formato del percorso non valido.")
    except ValueError:
        raise ValueError("Il percorso non contiene 'Health Damage Data'.")


def get_region_from_path(path):
    path_parts = path.split('/')

    # Trova l'indice della cartella 'Health Damage Data'
    try:
        health_damage_index = path_parts.index('Health Damage Data')
        # Restituisci l'elemento successivo dopo 'Health Damage Data'
        if health_damage_index + 2 < len(path_parts):
            return path_parts[health_damage_index + 2]
        else:
            raise ValueError("Formato del percorso non valido.")
    except ValueError:
        raise ValueError("Il percorso non contiene 'Health Damage Data'.")


def diff_data(d1, d2, time):
    dt1 = datetime.fromisoformat(d1)
    dt2 = datetime.fromisoformat(d2)

    # Calcola la differenza tra le due date
    differenza = abs(dt2 - dt1)
    if differenza > time:
        return 1
    elif differenza < time:
        return 2
    else:
        return 0


def interval_list_value(dimension, control_time):
    tmp = []
    x = int(dimension / (int(control_time / 5)))
    for i in range(0, x):
        tmp.append(int(control_time / 5))
    tmp.append(dimension - (x * int(control_time / 5)))
    return tmp


def sum_last_n_elements(array, n):
    return sum(array[-n:])


def data_initialization():
    return {
        'Date': [],
        'No Optimization': [],
        'Static Follow the Sun': [],
        'Flexible Follow the Sun 60': [],
        'Flexible Follow the Sun 120': [],
        'Flexible Start': [],
        'Pause and Resume 60': [],
        'Pause and Resume 120': [],

    }


def next_multiple_of_5(date_str):
    # Parse the date string into a datetime object
    date = datetime.fromisoformat(date_str)

    # Calculate the number of minutes to add to reach the next multiple of 5
    minutes_to_add = 5 - (date.minute % 5)
    if minutes_to_add == 5 and date.second > 0:
        minutes_to_add = 0

    # Set seconds to zero
    date = date.replace(second=0, microsecond=0)

    # Add the calculated minutes to the datetime
    next_date = date + timedelta(minutes=minutes_to_add)

    return next_date.isoformat()


def get_difference_in_minutes(date_str1, date_str2):
    # Converti le stringhe in oggetti datetime
    date_format = "%Y-%m-%dT%H:%M:%S%z"
    date1 = datetime.strptime(date_str1, date_format)
    date2 = datetime.strptime(date_str2, date_format)

    # Calcola la differenza
    difference = date2 - date1  # La differenza avrà il segno appropriato

    # Estrai i minuti dalla differenza
    difference_in_minutes = difference.total_seconds() / 60

    return difference_in_minutes
