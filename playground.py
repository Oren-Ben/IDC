from first_ex import *

# data = create_csv(cols,num_of_rows)

conn = create_connection('mydata.db')
# print(conn)

# create_table(conn, create_table_query)

# insert_rows(conn,'mydata.csv')

# mydata = retrieve_data(conn, select_all_query)

# print_rows(mydata)


# apples_above_90 = retrieve_data(conn,select_all_apples_with_price_above_90)

# print_rows(apples_above_90)
# blue_oranges = retrieve_data(conn,select_all_blue_oranges)

# columns = retrieve_data(conn,query)
# print_rows(columns)

# df_pa = read_csv_with_pyarrow(mydata_csv)

# df = df_pa.to_pandas()
# print(df.head())

# print(count_rows(df_pa))

# crete_parquet_with_pyarrow(mydata_csv)

# pa_parquet = pd.read_parquet('mydatapyarrow.parquet')

# print(pa_parquet.head())


# crete_parquet_with_dask(mydata_csv)

# dd_parquet = dd.read_parquet('mydatadask.parquet')

# print(dd_parquet.head())

# crete_parquet_with_pandas(mydata_csv)

# df_parquet = pd.read_parquet('mydatapandas.parquet')

# print(df_parquet.head())

# get_size_csv_file(mydata_csv)


#print(first_chunk(mydata_csv, middle))

#print("-----------")

#print(last_chunk(mydata_csv, middle))
print(split_to_chunks(mydata_csv,16000000))


#print(mg16_chunks())

#print(sum_of_rows)

# print(last_chunk())
