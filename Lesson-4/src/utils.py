import numpy as np
from scipy.sparse import csr_matrix


def prefilter_items(data_train, cut_length=3, sales_range=(10, 100)):
    # Оставим только 5000 самых популярных товаров
    popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values('n_sold', ascending=False).head(
        5000).item_id.tolist()

    # добавим, чтобы не потерять юзеров
    data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999999
    # Уберем самые популярные
    # Уберем самые непопулряные
    top_5000_cut = top_5000[cut_length:-cut_length]
    data_train.loc[
        ~data_train['item_id'].isin(top_5000_cut), 'item_id'] = 999999

    # Уберем товары, которые не продавались за последние 12 месяцев
    data_train = data_train[
        data_train['week_no'] > data_train['week_no'].max() - 52]

    # Уберем не интересные для рекоммендаций категории (department)
    to_del = ['GROCERY', 'MISC. TRANS.', 'PASTRY',
              'DRUG GM', 'MEAT-PCKGD',
              'SEAFOOD-PCKGD', 'PRODUCE',
              'NUTRITION', 'DELI', 'COSMETICS']
    data_train = data_train[~(data_train['department'].isin(to_del))]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data_train = data_train[~(data_train['sales_value'] < sales_range[0])]

    # Уберем слишком дорогие товарыs
    data = data_train[~(data_train['sales_value'] > sales_range[1])]

    return data
