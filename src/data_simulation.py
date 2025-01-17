"""
Python script for simulating the data from the 3 events
"""
import polars as pl
import numpy as np
import names
import logging
import random
import seaborn as sns
import matplotlib.pyplot as plt
from polars.testing import assert_frame_equal
from datetime import datetime, timedelta
from pathlib import Path
from utils import init_logger


WINE_DATA = [
    {'id': 1, 'brand': 'Langhe Nebbiolo Rosa dell Olmo 2021', 'size -ml': 750, 'type': 'red', 'cost': 8.99},
    {'id': 2, 'brand': 'Motif Cabernet Sauvignon Red Hill', 'size -ml': 750, 'type': 'red', 'cost': 4.99},
    {'id': 3, 'brand': 'Chianti Classico', 'size -ml': 1500, 'type': 'sparkling', 'cost': 19.99},
    {'id': 4, 'brand': 'Prosecco', 'size -ml': 750, 'type': 'sparkling', 'cost': 8.99},
    {'id': 5, 'brand': 'Prosecco', 'size -ml': 750, 'type': 'sparkling', 'cost': 8.99},
    {'id': 6, 'brand': 'Giardino Pinot Grigio', 'size -ml': 1500, 'type': 'white', 'cost': 14.99},
    {'id': 7, 'brand': 'Honey Moon', 'size -ml': 750, 'type': 'white', 'cost': 5.49}

]

WINE_IDS = [x['id'] for x in WINE_DATA]
WINE_1_PROBS = [0.099999999999999907, 0.18, 0.12, 0.13, 0.15, 0.14, 0.180000000000000009]
WINE_2_PROBS = [0.099999999999999907, 0.07, 0.08, 0.13, 0.200000000000000009, 0.24, 0.180000000000000009]
WINE_3_PROBS = [0.34, 0.21, 0.25, 0.09, 0.01, 0.01, 0.09]

EVENT_DRINKS_PER_PERSON = [0, 1, 2, 3, 4, 5, 6] # The higheat number of drinks a person had is 6

EVENT_1_DRINK_PROBS = [0.03, 0.12, 0.22, 0.43, 0.15, 0.039999999999999925, 0.01] # Event 1, 22% of attendants had 2 drinks
EVENT_2_DRINK_PROBS = [0.02, 0.15, 0.45, 0.24, 0.08, 0.050000000000000027, 0.01]
EVENT_3_DRINK_PROBS = [0.03, 0.11, 0.37, 0.33, 0.11, 0.03, 0.019999999999999907]

EVENT_1_ATTENDEES = 36
EVENT_2_ATTENDEES = 52
EVENT_3_ATTENDEES = 71


def get_attendee_first_names(num_f: int = 10,
                             num_m: int = 10) -> list[str]:
    """
    Return a list or tuple of first names and last names if specified
    :param num_f: int
        the number of female first names
    :param num_m: int
        the number of male first names
    :param last_name: bool
        return a list of last names the size of num_f + num_m
    :return:
    """
    return [names.get_first_name(gender='female') for _ in range(num_f)] + [
        names.get_first_name(gender='male') for _ in range(num_m)]


def get_attendee_last_names(n: int) -> list[str]:
    """
    Return a list of n last names
    :param n: int
        the number of last names to return
    :return:
    """
    return [names.get_last_name() for _ in range(n)]


def discrete_sampler(vals: list, n: int, probs: list[float]) -> np.array:
    """

    :param vals:
    :param n:
    :param probs:
    :return:
    """
    return np.random.choice(vals, n, p=probs)


def map_customers_to_orders(df_counts: pl.DataFrame,
                            id_col: str = 'First name',
                            val_col: str = 'Drink Count') -> list:
    """
    Map customers to orders based on the aggregated counts of drinks each person ordered at the event
    :param df_counts: pl.DataFrame
        The dataframe of drinks odered by each attendee
    :param id_col: str
        The column that uniquely identifies each row
    :param val_col: str
        The column representing the counts
    :return: pl.DataFrame
    """
    count_dict = df_counts.select([id_col, val_col]).to_dict(as_series=False)
    customer_list = []
    for id, val in zip(count_dict[id_col], count_dict[val_col]):
        customer_list = customer_list + [id for _ in range(val)]

    random.shuffle(customer_list)

    return customer_list





if __name__ == "__main__":

    init_logger(file_name='simulation_data.log')
    logging.info(f"Create the wine data...")
    wine_data = pl.DataFrame(WINE_DATA)
    # check the file path exists, if not create one
    data_path = Path("data")
    if not data_path.exists():
        data_path.mkdir(parents=True)
    wine_data.write_parquet("data/wine.parq")
    logging.info("Wine data has been persisted: data/wine.parq")
    logging.info(wine_data.head())

    # simulate the number of drinks per person for the three events
    attendees_first_name = get_attendee_first_names(num_f=20, num_m=16)
    attendees_first_name2 = get_attendee_first_names(num_f=30, num_m=22)
    attendees_first_name3 = get_attendee_first_names(num_f=40, num_m=31)

    event_1_drink_counts = discrete_sampler(vals=EVENT_DRINKS_PER_PERSON, n=EVENT_1_ATTENDEES, probs=WINE_1_PROBS)
    event_2_drink_counts = discrete_sampler(vals=EVENT_DRINKS_PER_PERSON, n=EVENT_2_ATTENDEES, probs=WINE_2_PROBS)
    event_3_drink_counts = discrete_sampler(vals=EVENT_DRINKS_PER_PERSON, n=EVENT_3_ATTENDEES, probs=WINE_3_PROBS)

    # create the dataframes
    df_drink_counts = pl.DataFrame({
        'cust_id': [x for x in range(EVENT_1_ATTENDEES)],
        'First Name': attendees_first_name,
        'Last Name': get_attendee_last_names(n=EVENT_1_ATTENDEES),
        'Drink Count': event_1_drink_counts,
        'event number': [1 for _ in range(EVENT_1_ATTENDEES)]
    })

    df_drink_counts2 = pl.DataFrame({
        'cust_id': [x + 500 for x in range(EVENT_2_ATTENDEES)],
        'First Name': attendees_first_name2,
        'Last Name': get_attendee_last_names(n=EVENT_2_ATTENDEES),
        'Drink Count': event_2_drink_counts,
        'event number': [2 for _ in range(EVENT_2_ATTENDEES)]
    })

    df_drink_counts3 = pl.DataFrame({
        'cust_id': [x + 1000 for x in range(EVENT_3_ATTENDEES)],
        'First Name': attendees_first_name3,
        'Last Name': get_attendee_last_names(n=EVENT_3_ATTENDEES),
        'Drink Count': event_3_drink_counts,
        'event number': [3 for _ in range(EVENT_3_ATTENDEES)]
    })

    # concat the three dataframe
    df_drink_counts = pl.concat([df_drink_counts, df_drink_counts2, df_drink_counts3])
    logging.info("Create the aggregated drink counter per person, writing data ...")
    df_drink_counts.write_parquet("data/drink_counts.parq")
    logging.info("drink counts per person at each event persisted: data/drink_counts.parq")
    logging.info(df_drink_counts.head())

    # simulate the frequency of drinks ordered based on the total amount of drinks orderd

    # get the total number of drinks ordered
    order_count = int(event_1_drink_counts.sum())
    order_count2 = int(event_2_drink_counts.sum())
    order_count3 = int(event_3_drink_counts.sum())
    logging.info(f"Order counts: {order_count}, {order_count2}, {order_count3}")

    logging.info(f"wine ids: {WINE_IDS}\nlength: {len(WINE_IDS)}")
    logging.info(f"size of probs array\n1: {len(WINE_1_PROBS)}\n2: {len(WINE_2_PROBS)}\n3: {len(WINE_3_PROBS)}")
    event_1_drink_counts = discrete_sampler(vals=WINE_IDS, n=order_count, probs=WINE_1_PROBS)
    event_2_drink_counts = discrete_sampler(vals=WINE_IDS, n=order_count2, probs=WINE_2_PROBS)
    event_3_drink_counts = discrete_sampler(vals=WINE_IDS, n=order_count3, probs=WINE_3_PROBS)

    event_1_date_list = [datetime(year=2023, month=11, day=16, hour=9, minute=0) + timedelta(minutes=x) for x in
                         range(order_count)]
    event_2_date_list = [datetime(year=2024, month=5, day=28, hour=9, minute=0) + timedelta(minutes=x) for x in
                         range(order_count2)]
    event_3_date_list = [datetime(year=2024, month=11, day=18, hour=9, minute=0) + timedelta(minutes=x) for x in
                         range(order_count3)]

    logging.info(f"Order counts: {order_count}, {order_count2}, {order_count3}")
    # create the dataframes
    df_wine_orders = pl.DataFrame({
        'order_date': event_1_date_list,
        'id': event_1_drink_counts,
        'event number': [1 for _ in range(order_count)],
        'order_id': [x for x in range(len(event_1_drink_counts))],
        'cust_id': map_customers_to_orders(df_counts=df_drink_counts.filter(pl.col('event number') == 1),
                                           id_col='cust_id',
                                           val_col='Drink Count')
    })

    df_wine_orders2 = pl.DataFrame({
        'order_date': event_2_date_list,
        'id': event_2_drink_counts,
        'event number': [2 for _ in range(order_count2)],
        'order_id': [500 + x for x in range(len(event_2_drink_counts))],
        'cust_id': map_customers_to_orders(df_counts=df_drink_counts2.filter(pl.col('event number') == 2),
                                           id_col='cust_id',
                                           val_col='Drink Count')
    })

    df_wine_orders3 = pl.DataFrame({
        'order_date': event_3_date_list,
        'id': event_3_drink_counts,
        'event number': [3 for _ in range(order_count3)],
        'order_id': [1100 + x for x in range(len(event_3_drink_counts))],
        'cust_id': map_customers_to_orders(df_counts=df_drink_counts3.filter(pl.col('event number') == 3),
                                           id_col='cust_id',
                                           val_col='Drink Count')
    })

    df_wine_orders = pl.concat([df_wine_orders, df_wine_orders2, df_wine_orders3])
    logging.info("created wine orders, persisting...")

    # check if the orders is the same as the aggreate counts
    df_counts_check = df_wine_orders.select(pl.col('cust_id')
                                            .value_counts(sort=False, name="Drink Count")).unnest('cust_id')
    # df_counts_check = df_wine_orders.group_by(['cust_id'])
    # sort
    df_drink_counts_filter = df_drink_counts.filter(pl.col('Drink Count') > 0)
    df_counts_check = df_counts_check.sort('cust_id')
    df_counts_check = df_counts_check.cast({'Drink Count': pl.Int64})
    df_drink_counts_filter = df_drink_counts_filter.sort('cust_id')
    df_drink_counts_filter = df_drink_counts_filter.cast({'Drink Count': pl.Int64})

    try:
        assert_frame_equal(df_counts_check.select(['cust_id', 'Drink Count']),
                          df_drink_counts_filter.select(['cust_id', 'Drink Count']))
        logging.info("orders matches aggreate counts per drink, persisting orders ...")
        df_wine_orders.write_parquet("data/orders.parq")
        logging.info("orders data written: data/orders.parq")
        logging.info(df_wine_orders.head())
    except Exception as e:
        logging.error("Check process and variables in the process, there is a mis-match in the data for orders")
        raise ValueError(f"Mis-match in data for orders and aggregates\n{e}")











