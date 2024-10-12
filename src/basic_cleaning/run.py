#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info(f"Running basic_cleaning with parameters: {args}")
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    
    logger.info("Loading artifact to dataframe with path")
    df = pd.read_csv(artifact_path)   
    
    logger.info("Cleaning the data frame") 
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    idx = df['longitude'].between(-74.250 , -73.500 ) \
                & df['latitude'].between(40.50, 41.20)
    df = df[idx].copy()

    
    filename = "clean_data"
    df.to_csv(filename, index=False)
    
    logger.info("Creating artifact save")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)
    
    os.remove(filename)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum number for price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum number for price",
        required=True
    )

    args = parser.parse_args()

    go(args)

