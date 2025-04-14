from mpi4py import MPI
import numpy as np
import time

def caesar_encrypt(message, step):
    result = ""
    for c in message:
        if c.isalpha():
            ascii_offset = 65 if c.isupper() else 97
            result += chr((ord(c) - ascii_offset + step) % 26 + ascii_offset)
        else:
            result += c
    return result

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# === Послідовне шифрування (тільки у процесі з rank 0) ===
if rank == 0:
    with open("wordsToEncrypt.txt", "r") as file:
        full_text = file.read()

    seq_start = time.time()
    sequential_result = caesar_encrypt(full_text, 3)
    seq_end = time.time()

    print("Sequential time:", seq_end - seq_start, "seconds")

    # Результат можна зберегти за бажанням
    with open("SequentialEncrypted.txt", "w") as f:
        f.write(sequential_result)

# === Паралельне шифрування з MPI ===
mpi_start_time = MPI.Wtime()

if rank == 0:
    text = list(full_text)
    chunks = np.array_split(np.array(text), size)
    chunks = [np.array(chunk) for chunk in chunks]
else:
    chunks = None

# Розсилаємо шматки тексту всім процесам
chunk = comm.scatter(chunks, root=0)

# Кожен процес шифрує свій шматок
encrypted_chunk = caesar_encrypt(chunk.tobytes().decode('utf-8').replace('\x00',''), 3)

# Збираємо результат назад на процес 0
result = comm.gather(encrypted_chunk, root=0)

if rank == 0:
    result = ''.join(result)
    with open("MPIEncrypted.txt", "w") as output_file:
        output_file.write(result)
    mpi_end_time = MPI.Wtime()
    print("MPI parallel time:", mpi_end_time - mpi_start_time, "seconds")