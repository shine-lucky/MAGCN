import numpy as np
import torch as t
import random

def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj



def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def load_miRNA_disease():
    print("Loading miRNA-disease data set...")
    filePath = '../Data/miRNA-disease_id.txt'
    miRNA_disease = np.zeros((268,799),dtype=np.int32)
    with open(filePath, 'r') as f:
        for line in f:
            if line:
                lines = line[:-1].split("::")
                m = int(lines[0])
                d = int(lines[1])
                miRNA_disease[m-1][d-1] = 1
    return miRNA_disease
def load_incRNA_microRNA():
    print("Loading incRNA_microRNA data set...")
    filePath = '../Data/lncRNA_miRNA_id.txt'
    IncRNA_miRNA = np.zeros((541,268),dtype=np.int32)
    with open(filePath, 'r') as f:
        for line in f:
            if line:
                lines = line[:-1].split("::")
                m = int(lines[0])
                d = int(lines[1])
                IncRNA_miRNA[m-1][d-1] = 1
    return IncRNA_miRNA



