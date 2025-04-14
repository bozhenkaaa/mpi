from mpi4py import MPI
import numpy as np

def caesar_encrypt(message, step):
    result = ""
    for c in message:
        if c.isalpha():
            ascii_offset = 65 if c.isupper() else 97
            result += chr((ord(c) - ascii_offset + step) % 26 + ascii_offset)
        else:
            result += c
    return result

comm = MPI.COMM_WORLD   # Initialize communicator
rank = comm.Get_rank()  # Rank of the current process
size = comm.Get_size()  # Total number of processes

start_time = MPI.Wtime()

# Only rank 0 process reads and manipulates the data
if rank == 0:
    # Data read from file
    with open("wordsToEncrypt.txt", "r") as file:
        text = list(file.read())

    chunks = np.array_split(np.array(text), size)
    chunks = [np.array(chunk) for chunk in chunks]
else:
    chunks = None

# Scatter chunks
chunk = comm.scatter(chunks, root=0)

# Each process encrypts its chunk
encrypted_chunk = caesar_encrypt(chunk.tobytes().decode('utf-8').replace('\x00',''), 3)

# Gather all encrypted chunks at root
result = comm.gather(encrypted_chunk, root=0)

if rank == 0:
    result = ''.join(result)
    with open("MPIEncrypted.txt", "w") as output_file:
        output_file.write(result)
    end_time = MPI.Wtime()
    print('Time taken:', end_time - start_time)