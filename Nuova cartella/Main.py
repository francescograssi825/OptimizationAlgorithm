import datetime

from ElaborateCsv import count_rows_csv_df, save_results
from GPU.Tools import Variables
from GPU.Optimization import Optimization
from GPU.Tools.csvUtils import get_csv
from Utils import add_minutes_to_date, extract_month


def main():
    data_center = ['aws', 'azure', 'gcp']
    #consumption_file = ['svm', 'autoencoder_corrected', 'new_hf_sca_consumption_5', 'isolation_forrest']
    consumption_file = ['new_hf_sca_consumption_5']

    day = 365
    date_now = datetime.datetime.now()

    for dc in data_center:
        print('data center ', dc)
        date_now_dc = datetime.datetime.now()
        for cs_file in consumption_file:
            print('consumption file ', cs_file)
            date_now_cs = datetime.datetime.now()

            __data_no_optimization = {'Date': [], '__data_no_optimization_': []}
            __data_flexible_follow_sun = {'Date': [], '__data_flexible_follow_sun_': [], 'time_window': [],
                                          'checking_time': [], 'number_of_change': [], 'history_change': []}
            __data_static_follow_sun = {'Date': [], '__data_static_follow_sun_': [], 'checking_time': [],
                                        'number_of_change': [], 'history_change': []}
            __data_flexible_start = {'Date': [], '__data_flexible_start_': [], 'data_flexible_start_region': [],
                                     'time_window': []}
            __data_pause_resume = {'Date': [], '__data_pause_resume_': [], 'data_pause_resume_region': [],
                                   'time_window': [], 'checking_time': []}

            df_cons = get_csv(rf'E:\uni\Python\OptimizationAlgor\csv_dir\{cs_file}_incremental.csv', None, 0)
            len_df_cons = count_rows_csv_df(df_cons)
            time = '2021-01-01T09:00:00+00:00'
            opt = Optimization(df_cons=df_cons, dc=dc, month=extract_month(time))

            for _ in range(day):
                date_now_day = datetime.datetime.now()
                x = opt.impact_no_optimization(time)
                __data_no_optimization['Date'].append(time)
                __data_no_optimization['__data_no_optimization_'].append(x / 1000)

                y = opt.simulate_flexible_start(time)

                for key in list(y.keys()):
                    for i in range(0, 4):
                        __data_flexible_start['Date'].append(time)
                        __data_flexible_start['data_flexible_start_region'].append(key)
                        __data_flexible_start['__data_flexible_start_'].append(y[key][i] / 1000)
                        __data_flexible_start['time_window'].append(Variables.GlobalVariables.time_window[i] / 60)
                    for i in range(4, 8):
                        __data_flexible_start['Date'].append(time)
                        __data_flexible_start['data_flexible_start_region'].append(key)
                        __data_flexible_start['__data_flexible_start_'].append(y[key][i] / 1000)
                        __data_flexible_start['time_window'].append(
                            Variables.GlobalVariables.time_window_percentage[i - 4])

                z = opt.simulate_static_start_follow_the_sun(time)

                for key in list(z.keys()):
                    __data_static_follow_sun['Date'].append(time)
                    __data_static_follow_sun['__data_static_follow_sun_'].append(z[key]['product'] / 1000)
                    __data_static_follow_sun['checking_time'].append(z[key]['interval'] * 5)
                    __data_static_follow_sun['number_of_change'].append(len(list(z[key]['change'])))
                    __data_static_follow_sun['history_change'].append(list(z[key]['change']))

                m = opt.simulate_flexible_start_follow_the_sun(time)

                for key in list(m.keys()):

                    for k in list(m[key].keys()):

                        if int(key) * 5 in Variables.GlobalVariables.time_window:
                            __data_flexible_follow_sun['time_window'].append(int(key) * 5)
                        else:
                            __data_flexible_follow_sun['time_window'].append(int((int(key) * 100) / len_df_cons))
                        __data_flexible_follow_sun['checking_time'].append(m[key][k]['interval'] * 5)
                        __data_flexible_follow_sun['Date'].append(time)
                        __data_flexible_follow_sun['number_of_change'].append(len(list(m[key][k]['change'])))
                        __data_flexible_follow_sun['history_change'].append(list(m[key][k]['change']))
                        __data_flexible_follow_sun['__data_flexible_follow_sun_'].append(m[key][k]['product'] / 1000)

                p_a_r = opt.simulate_pause_and_resume(time)

                for key in p_a_r.keys():
                    for k in p_a_r[key].keys():
                        for c in p_a_r[key][k].keys():
                            if int(k) * 5 in Variables.GlobalVariables.time_window:
                                __data_pause_resume['time_window'].append(int(k) * 5)
                            else:
                                __data_pause_resume['time_window'].append(int((int(k) * 100) / len_df_cons))
                            __data_pause_resume['Date'].append(time)
                            __data_pause_resume['data_pause_resume_region'].append(key)
                            __data_pause_resume['checking_time'].append(c)
                            __data_pause_resume['__data_pause_resume_'].append(p_a_r[key][k][c])

                print("Durata giorno ", time, " : ", datetime.datetime.now() - date_now_day)

                if extract_month(add_minutes_to_date(time, 60 * 24)) > extract_month(time) or (extract_month(time) == 12 and extract_month(add_minutes_to_date(time, 60 * 24)) == 1):
                    print('end month, start csv saving')
                    save_results(time, __data_no_optimization, cs_file, opt.dc)
                    save_results(time, __data_static_follow_sun, cs_file, opt.dc)
                    save_results(time, __data_flexible_follow_sun, cs_file, opt.dc)
                    save_results(time, __data_flexible_start, cs_file, opt.dc)
                    save_results(time, __data_pause_resume, cs_file, opt.dc)
                    opt = Optimization(df_cons=df_cons, dc=dc, month=extract_month(add_minutes_to_date(time, 60 * 24)))

                time = add_minutes_to_date(time, 60 * 24)
                if time == "2022-01-01T09:00:00+00:00":
                    time = "2021-01-01T09:00:00+00:00"

            print('Tempo di esecuzione per il file ', cs_file, 'nel data center ', dc, ': ',
                  datetime.datetime.now() - date_now_cs)
        print("Tempo di esecuzione per il data center ", dc, ": ", datetime.datetime.now() - date_now_dc)
    print('Tempo di esecuzione: ', datetime.datetime.now() - date_now)


if __name__ == '__main__':
    main()
