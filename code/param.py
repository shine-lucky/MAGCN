import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MMGCN.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="../MDA2.0",
                        help="Training datasets.")

    parser.add_argument("--epoch",
                        type=int,
                        default=200,
                        help="Number of training epochs. Default is 651.")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="random seed is 1.")
    parser.add_argument("--k_fold",
                        type=int,
                        default=5,
                        help="k_fold is 5.")
    parser.add_argument("--alpha",
                        type=int,
                        default=0.0001,
                        help="alpha.")
    parser.add_argument("--learn_rate",
                        type=int,
                        default=0.001,
                        #default=0.001,
                        help="learning rate. Default is 0.001.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out_channels",
                        type=int,
                        default=128,
                        help="out-channels of cnn. Default is 128.")

    parser.add_argument("--mirna_size",
                        type=int,
                        default=268,
                        help="miRNA number. Default is 256.")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.00005,
                        help='Weight decay.')

    parser.add_argument("--f0",
                        type=int,
                        default=512,
                        help="miRNA number. Default is 512.")

    parser.add_argument("--disease_size",
                        type=int,
                        default=799,
                        help="disease number. Default is 799.")

    parser.add_argument("--lncrna_size",
                        type=int,
                        default=541,
                        help="disease number. Default is 799.")



    parser.add_argument("--view",
                        type=int,
                        default=2,
                        help="views number. Default is 2(2 datasets for miRNA and disease sim)")


    return parser.parse_args()