#! /usr/bin/env python

import sys


def readGenome(filename):
    genome = ''
    with open(filename, 'r') as f:
        for line in f:
            # ignore header line with genome information
            if not line[0] == '>':
                genome += line.rstrip()
    return genome


def editDistanceAprox(x, y):
    # Create distance matrix
    D = []
    for i in range(len(x)+1):
        D.append([0]*(len(y)+1))
    # Initialize first row and column of matrix
    for i in range(len(x)+1):
        D[i][0] = i
    # For approximate matching, we se the rows to 0
    for i in range(len(y)+1):
        D[0][i] = 0
    # Fill in the rest of the matrix
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            distHor = D[i][j-1] + 1
            distVer = D[i-1][j] + 1
            if x[i-1] == y[j-1]:
                distDiag = D[i-1][j-1]
            else:
                distDiag = D[i-1][j-1] + 1
            D[i][j] = min(distHor, distVer, distDiag)
    return min(D[-1][:]), D


def traceback(D, x, y):
    i = len(D[:]) - 1
    j = D[-1].index(min(D[-1][:]))
    alig_A = ""
    alig_B = ""
    while (i > 0 and j > 0):
        if (x[i-1] == y[j-1]):
            overlap = 0
        else:
            overlap = 1
        if (i > 0 and j > 0 and (D[i][j] == (D[i-1][j-1] + overlap))):
            alig_A = x[i-1] + alig_A
            alig_B = y[j-1] + alig_B
            i = i - 1
            j = j - 1
        elif (i > 0 and (D[i][j] == (D[i-1][j] + 1))):
            alig_A = x[i-1] + alig_A
            alig_B = "-" + alig_B
            i = i - 1
        else:
            alig_A = "-" + alig_A
            alig_B = y[j-1] + alig_B
            j = j - 1
    return alig_A, alig_B

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Feed me a fasta file ")
        sys.exit()
    else:
        print("Assuming you fed me a fasta, expect garbage otherwise ")
        gen = "AGGGGCTCGCAGTGTAAGAA"
        gen = readGenome(sys.argv[1])
        pp = "AGTGTCAACAGGCAATTATCTTCCTGGG"
        p = "GCGTATGC"
        t = "TATTGGCTATACGGTT"

        ppp = "TCGGTAGATTGCGCCCACTC"

        edit_d, D_mat = editDistanceAprox(ppp, gen)
        print("Edit distance is ", edit_d)
        ali_A, ali_B = traceback(D_mat, ppp, gen)
        print("ppp", ppp)
        print("gen", gen)
        print("Align A", ali_A)
        print("Align B", ali_B)
