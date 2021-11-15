import os

FILE_NAME = 'mydata.csv'
file_size = os.path.getsize(FILE_NAME)
middle = file_size//2
chunk_size = 16000000


def first_chunk(file_name,size_of_chunk):
    with open(file_name,'rb') as file:
        return len(file.readlines(size_of_chunk))

def last_chunk(file_name,size_of_chunk):
    with open(file_name, 'rb') as file:
        file.seek(size_of_chunk+1,0)
        return len(file.readlines()[1:])

def read_chunk(file_name,start,size_of_chunk):
    with open(file_name, 'rb') as file:
        file.seek(start+1,0)
        return len(file.readlines(size_of_chunk)[1:])

# print(first_chunk(FILE_NAME,middle))
# print(last_chunk(FILE_NAME,middle))
print(first_chunk(FILE_NAME,middle)+last_chunk(FILE_NAME,middle))

def algo(file_name,chunk_size):
    rows_counter = []
    rows_counter.append(first_chunk(file_name,chunk_size))
    for byte in range(chunk_size,file_size,chunk_size):
        rows_counter.append(read_chunk(file_name,byte,chunk_size))
    return rows_counter

print(sum(algo(FILE_NAME,chunk_size)))

