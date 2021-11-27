Oren_id = 204079453
Yuval_id = 205714447

import csv
import os
import sqlite3
from sqlite3 import Error

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq


# In general, all our print commands, will appear at the bottom of the page.

# Q1.1
def create_csv(cols_name, num_of_rows):
    data_dict = {
        cols_name[0]: np.arange(1, num_of_rows + 1),
        cols_name[1]: np.random.choice(fruit_choice, num_of_rows),
        cols_name[2]: np.random.randint(10, 100, num_of_rows),
        cols_name[3]: np.random.choice(color_choice, num_of_rows)
    }
    df = pd.DataFrame(data_dict)
    return df.to_csv('mydata.csv', index=False)


num_of_rows = 10000000
cols = ['id', 'fruit', 'price', 'color']
fruit_choice = ['Orange', 'Grape', 'Apple', 'Banana', 'Pineapple', 'Avocado']
color_choice = ['Red', 'Green', 'Yellow', 'Blue']


def create_connection(db_data):
    conn = None
    try:
        conn = sqlite3.connect(db_data)
    except Error as e:
        print(e)
    finally:
        return conn


# Q1.2
create_table_query = '''CREATE TABLE IF NOT EXISTS mydata(
                id integer PRIMARY KEY,
                fruit text,
                price integer,
                color text);
                '''


def create_table(conn, create_table_query):
    try:
        c = conn.cursor()
        c.execute(create_table_query)
    except Error as e:
        print(e)


records_query = '''INSERT INTO mydata (id,fruit, price, color) VALUES(?, ?, ?, ?)'''


def insert_rows(conn, file_name):
    try:
        c = conn.cursor()
        # Opening the mydata-records.csv file
        file = open(file_name)
        # Reading the contents of the
        contents = csv.reader(file)
        next(contents, None)
        insert_records = records_query
        
        # SQL query to insert data into the
        # Importing the contents of the file
        # into our mydata table
        c.executemany(insert_records, contents)
        
        # Committing the changes
        conn.commit()
    
    except Error as e:
        print(e)


select_all_query = "SELECT * FROM mydata;"

query = 'pragma table_info(mydata)'


def retrieve_data(conn, select_query):
    try:
        c = conn.cursor()
        select_all = select_query
        rows = c.execute(select_all).fetchall()
        # Output to the console screen
        return rows
    
    except Error as e:
        print(e)


# Q1.3

# The query retrieve the rows that meet the criteria of fruit is Apple and price above 90
# The predicate operation is the where statement and the projection is at the select part as id
select_all_apples_with_price_above_90 = " SELECT id FROM mydata Where fruit = 'Apple' and price > 90"

# The query retrieve the rows that meet the criteria of fruit is Orange and color is Blue
# The predicate operation is the where statement and the projection is at the select part as * -> meaning all columns
select_all_blue_oranges = " SELECT * FROM mydata Where fruit = 'Orange' and color = 'Blue'"


# Q2.1
def read_csv_with_pyarrow(file_csv):
    df_pa = pa_csv.read_csv(file_csv)
    return df_pa


def count_rows(df_pa):
    return df_pa.num_rows


# Q2.2
def crete_parquet_with_pyarrow(file_csv):
    table = read_csv_with_pyarrow(file_csv)
    pq.write_table(table, 'mydatapyarrow.parquet')


# Q2.3
def crete_parquet_with_dask(file_csv):
    df = dd.read_csv(file_csv)
    df.to_parquet('mydatadask.parquet')


# Q2.4
def crete_parquet_with_pandas(file_csv):
    df = pd.read_csv(file_csv)
    df.to_parquet('mydatapandas.parquet')


# Q2.5
# PyArrow and pandas store data in a columnar method.
# Apache Arrow takes advantage of a columnar buffer to reduce IO and accelerate analytical processing performance.
# Dask is a library for parallel computation,
# Dask DataFrame is split up into many Pandas DataFrames that sometimes call “partitions"
# to scale it to multi-core machines and distributed clusters.

# Q3.1
def get_size_csv_file(file_csv, print_cond=False):
    file_size = os.path.getsize(file_csv)
    if not print_cond:
        print(f"File Size is :{file_size} bytes")
    return file_size


def first_chunk(file_csv, middle):
    f1 = open(file_csv, 'rb')
    f1.seek(0)
    d1 = f1.read(middle).decode(encoding='utf-8')
    rows = d1.splitlines()
    return len(rows)


def last_chunk(file_csv, middle):
    f2 = open(file_csv, 'rb')
    f2.seek(middle + 1, 0)
    d2 = f2.read().decode(encoding='utf-8')
    rows = d2.splitlines()
    return len(rows)


# Q3.3

# In section 2.1, the number of lines is 10,000,000 +1header = 10,000,001 (including the header line).
# In this section, we received 10,000,002.
# The reason that cause the difference between those results:
# Is the middle row, it is divided into two, therefore counted
# once as a regular row and in the last chunk as a supplement.


# Q3.4

# We will first solve the original question,
# splitting into the first and last chunk based on the middle file size
# without "adding" another row

def first_chunk_update(file_name, size_of_chunk):
    with open(file_name, 'rb') as file:
        return len(file.readlines(size_of_chunk))


def last_chunk_update(file_name, size_of_chunk):
    with open(file_name, 'rb') as file:
        file.seek(size_of_chunk + 1, 0)
        return len(file.readlines()[1:])


# Following this, we will suggest another solution to split the data into chunks based on chunk size.

def read_chunk(file_name, start, size_of_chunk):
    with open(file_name, 'rb') as file:
        file.seek(start + 1, 0)
        return len(file.readlines(size_of_chunk)[1:])


def algo(file_name, chunk_size):
    file_size = get_size_csv_file(file_name, print_cond=True)
    rows_counter = []
    rows_counter.append(first_chunk_update(file_name, chunk_size))
    for byte in range(chunk_size, file_size, chunk_size):
        rows_counter.append(read_chunk(file_name, byte, chunk_size))
    return rows_counter


if __name__ == "__main__":
    MYDATA_CSV = 'mydata.csv'
    MYDATA_DB = 'mydata.db'
    
    # Intro -
    data = create_csv(cols, num_of_rows)
    
    # # Q1.1
    conn = create_connection(MYDATA_DB)
    create_table(conn, create_table_query)

    # Q1.2
    insert_rows(conn, MYDATA_CSV)
    my_data = retrieve_data(conn, select_all_query)
    
    # Q1.3
    apples_above_90 = retrieve_data(conn, select_all_apples_with_price_above_90)
    print(apples_above_90)
    blue_oranges = retrieve_data(conn, select_all_blue_oranges)
    print(blue_oranges)
    
    # Q2.1
    df_pa = read_csv_with_pyarrow(MYDATA_CSV)
    print(count_rows(df_pa))
    
    # Q2.2
    crete_parquet_with_pyarrow(MYDATA_CSV)
    pa_parquet = pd.read_parquet('mydatapyarrow.parquet')
    
    # Q2.3
    crete_parquet_with_dask(MYDATA_CSV)
    dd_parquet = dd.read_parquet('mydatadask.parquet')
    
    # Q2.4
    crete_parquet_with_pandas(MYDATA_CSV)
    df_parquet = pd.read_parquet('mydatapandas.parquet')
    
    # Q3.1
    get_size_csv_file(MYDATA_CSV)
    middle = get_size_csv_file(MYDATA_CSV, print_cond=True) // 2
    
    # Q3.2
    print(first_chunk(MYDATA_CSV, middle))
    print(last_chunk(MYDATA_CSV, middle))
    
    # Summing the total rows:
    sum_of_rows = first_chunk(MYDATA_CSV, middle) + last_chunk(MYDATA_CSV, middle)
    print(sum_of_rows)
    
    # Q3.5
    # Checking the first_chunk and last_chunk new algo:
    print(first_chunk_update(MYDATA_CSV, middle) + last_chunk_update(MYDATA_CSV, middle))
    # Checking the suggested algo on 16MB chunks:
    print(algo(MYDATA_CSV, 16000000))
