# This file is used to run parse_to_error_df from the command
# line with argparse.

import argparse
from polygnn_trainer.parse import parse_to_error_df

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_out_path", help="The path to load the training output from"
)
parser.add_argument("--group_map_path", help="The path to load the group map from")
parser.add_argument("--save_path", help="The path to save the ensemble metrics to")
args = parser.parse_args()

parse_to_error_df(args.train_out_path, args.group_map_path, args.save_path)
