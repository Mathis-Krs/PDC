# ############################################################################
# assignment.py
#           -
#           -
#           -
# ############################################################################

import numpy as np
from sys import path
path.append("")
import client, local_channel
import argparse
import string

number_of_bits = 6 #7 if args.architecture == "BPSK" else 8
separator = 1
interleaving = 1

# we create our own mapping from a subset of ascii printable characters to binary
ascii_printable = string.printable[:64]
binaries = [bin(number)[2:].zfill(number_of_bits) for number in range(len(ascii_printable))]
mapping = dict(zip(ascii_printable, binaries))

# creating the reverse mapping
reverse_mapping = dict((v, k) for k, v in mapping.items())


def transmitter(i: str, architecture: str) -> list:
    """
    reads a text message i and returns real-valued samples 
    of an information-bearing signal ci ∈ Rn, for some n ≤ 500,000
    """

    # creating codewords and encodings depending on the architecture
    if architecture == "BPSK":
        codewords = [1, -1]
        encoder = {
        "0": codewords[0],
        "1": codewords[1]
        }
    elif architecture == "4QAM":
        codewords = [separator, 2*separator, -separator, -2*separator]
        encoder = {
        "00": codewords[0],
        "01": codewords[1],
        "10": codewords[2],
        "11": codewords[3]
        }

    codewords = np.array(codewords, dtype=np.float64)

    

    # because the set A of possible messages i is of size 64, we can encode each message with 6 bits
    
    # we will use our custom mapping to encode the message
    encoded_message = []
    for char in i:
        encoded_message.append(mapping[char])
        

    # we need to map this binary sequence to a real-valued signal vector ci of length n (where n <= 500,000)
    # One common method for this is Binary Phase-Shift Keying (BPSK), where '0' is represented by a positive value (e.g., +1) and '1' is represented by a negative value (e.g., -1)

    # encoding the message
    if (architecture == "BPSK"):
        # we will use the BPSK method to encode the message
        ci = []
        for char in encoded_message:
            for b in char:
                ci += [encoder[b]]
        
    if (architecture == "4QAM"):
        # we will use the 4QAM method to encode the message
        ci = []
        for char in encoded_message:
            for i in range(0, len(char), 2):
                b = char[i:i+2]
                ci += [encoder[b]]

    ci = np.array(ci, dtype=np.float64)

    # trying interleaving
    ci = np.repeat(ci, interleaving)
        
    # we also need to ensure that the energy of the signal (||x||^2) is less than or equal to 40,960
    # This will constrain the choice of n and the amplitude of the BPSK signal

    # we will use the following formula to calculate the energy of the signal
    energy = np.linalg.norm(ci)**2

    # we amplify the signal at the maximum allowed value
    if architecture == "BPSK":
        # this is the amplification factor that sets the energy of the signal just below 40960
        amplification = np.sqrt(40960 / ci.size) - 10**(-4)
        ci *= amplification 
        
    else:
        amplification = 1

        while (energy < 40960):
            ci *= 2
            energy = np.linalg.norm(ci)**2
            amplification *= 2
        
    codewords *= amplification

    if energy > 40960:
        print(f"The energy of the signal is too high by {energy - 40960}")

    # we check that the length of the signal is less than or equal to 500,000
    if len(ci) > 500000:
        print("The length of the signal is too long")
        return
    
    return ci, codewords

    

def sender(ci: list, local=True) -> list:
    """
    Send ci to a server that applies the channel effect as in (1) and returns Y ∈ R2n
    """
    if local:
        return local_channel.channel(np.array(ci))

def receiver(Y: list, architecture:str, codewords: list[int]) -> str:
    """
    having received Y, reconstructs the text message with as few errors as possible
    """

    # The final step is to reconstruct the original text message from the received vector Y. This will involve first decoding the BPSK signal to obtain the 240-bit binary sequence, and then converting this binary sequence back to a sequence of 40 characters from the set A.
    # The decoding process will also need to take into account the noise Z introduced by the channel. One common method for this is Maximum Likelihood Decoding (MLD), where we choose the binary sequence that is most likely to have resulted in the received vector Y given the channel transformation and the noise distribution.

    # we will use the MLD method to decode the message

    # check exercise 4 for a more sophisticated decoding method/ML rule

    # de-interleaving the signal
    Y = np.reshape(Y, (-1, interleaving))
    Y = np.mean(Y, axis=1)

    # cutting Y in 2 parts
    n = Y.size
    y_1 = Y[:n//2]
    y_2 = Y[n//2:]

    decoded_message = []

    if architecture == "BPSK":
        decoder = {
            0: [0],
            1: [1]
        }

    elif architecture == "4QAM":
        decoder = {
            0: [0, 0],
            1: [0, 1],
            2: [1, 0],
            3: [1, 1]
        }

    # length is the number of bits that will be compared for decoding
    length = number_of_bits 

    import itertools
    # creating all codewords possibilities for length length
    codewords_full = np.array([list(e) for e in list(itertools.product(codewords, repeat=length))])
    
    # applying decision rule as in exercise 4.a of theory
    for i in range(0, len(y_1), length):
        test1 = (y_1[i:i+length].reshape(-1) - codewords_full)

        norm_1 = np.linalg.norm(y_1[i:i+length].reshape(-1) - codewords_full, axis=1)
        norm_2 = np.linalg.norm(y_2[i:i+length].reshape(-1) - codewords_full, axis=1)

        i_1 = np.argmin(norm_1, axis=0)
        i_2 = np.argmin(norm_2, axis=0)

        c_1 = codewords_full[i_1]
        c_2 = codewords_full[i_2]

        d_1 = np.linalg.norm(y_1[i:i+length] - c_1)**2 + np.linalg.norm(y_2)**2
        d_2 = np.linalg.norm(y_2[i:i+length] - c_2)**2 + np.linalg.norm(y_1)**2
        if d_1 < d_2:
            codeword = c_1
        else:
            codeword = c_2
        for c in codeword:
            decoded_message += decoder[np.argmin(np.abs(c - codewords))]


    # we will use our custom mapping to decode the message
    message = ""
    for i in range(0, len(decoded_message), number_of_bits):
        char = decoded_message[i:i+number_of_bits]
        char = "".join(map(str, char))

        message += reverse_mapping[char]

    return message

def main(args):

    file = open("message.txt", "w")

    message = args.message
    
    if not message:
        import string, random
        letters = ascii_printable
        message = ''.join(random.choices(letters, k=40))

    repeat = args.test
    if not repeat:
        repeat = 1
    counter = 0
    
    for _ in range(repeat):
        ci, codewords = transmitter(message, architecture=args.architecture)
        Y = sender(ci, local= not args.server)
        i = receiver(Y, architecture=args.architecture, codewords=codewords)
        test = "Test passed" if i == message else "Test failed"
        if i == message:
            counter += 1

    file.write(f"message: {message}\n")
    print(f"{message}")
    file.write(f"architecture used: {args.architecture}\n" + f"message: {message}\n")

    if repeat == 1: 
        file.write(str(ci) + "\n")
        file.write(f"signal's energy: {np.linalg.norm(ci)**2}\n")
        file.write(f"signal's length: {ci.size}\n")
        file.write(str(Y) + "\n")
        file.write(i)
        file.write(test)
        print(i)
        print(test)
    else:
        file.write(f"Success rate: {counter}/{repeat}")
        print(f"Success rate: {counter}/{repeat}")

    file.close()

if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description="COM-302 black-box channel simulator. (client)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="To enforce efficient communication schemes, transmissions are limited to 5e5 samples."
    )
    args.add_argument('--message', "-m", type=str, help="The message to be sent")
    args.add_argument('--server', "-s", default=False, action="store_true", help="Use server channel simulator")
    args.add_argument('--test', "-t", default=False, type = int, help="Automatically test the channel simulator, without entering a message")
    args.add_argument('--architecture', "-a", type=str, default="BPSK", help="choose architecure of the channel (BPSK, QPSK, 8PSK, 16QAM, 64QAM)")
    args = args.parse_args()
    main(args)