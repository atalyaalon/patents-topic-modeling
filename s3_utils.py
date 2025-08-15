"""Utility functions for loading different types of files to and from S3."""

from functools import cache
import io
import os
import logging
import pickle
import shutil
import tempfile
import numpy as np
import faiss
from bertopic import BERTopic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import boto3
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
S3_BUCKET = os.environ["S3_BUCKET"]
DATASET_TYPE = os.environ["DATASET_TYPE"]
S3_PREFIX = f"outputs_{str.lower(DATASET_TYPE)}"
SAVE_DIR = S3_PREFIX

def get_s3_client():
    """Get or create S3 client."""
    return boto3.client('s3')

def upload_to_s3(local_path: str, bucket: str = S3_BUCKET, prefix: str = S3_PREFIX) -> None:
    """Upload a local file to an S3 bucket.
    
    Args:
        local_path (str): Path to the local file.
        bucket (str): S3 bucket name.
        s3_prefix (str): Prefix (folder path) in the S3 bucket.
    
    Raises:
        Exception: If the upload fails.
    """
    filename = os.path.basename(local_path)
    s3_key = f"{prefix}/{filename}"
    try:
        s3 = get_s3_client()
        s3.upload_file(local_path, bucket, s3_key)
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        raise

def load_pickle_from_s3(key: str, bucket: str = S3_BUCKET, prefix: str = S3_PREFIX) -> Any:
    """Load a pickle file from S3.
    
    Args:
        key (str): The name of the file to load
        bucket (str): The S3 bucket name
        prefix (str, optional): The prefix (folder path) in the bucket
        
    Returns:
        Any: The unpickled object
        
    Raises:
        Exception: If there's an error loading the file
    """
    s3_path = f"{prefix}/{key}" if prefix else key
    logger.info(f"Loading pickle file from s3://{bucket}/{s3_path}")
    
    try:
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_path)
        return pickle.load(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        logger.error(f"Error loading pickle file from S3: {str(e)}")
        raise

def load_numpy_from_s3(key: str, bucket: str = S3_BUCKET, prefix: str = S3_PREFIX) -> np.ndarray:
    """Load a numpy array from S3.
    
    Args:
        key (str): The name of the .npy file to load
        bucket (str): The S3 bucket name
        prefix (str, optional): The prefix (folder path) in the bucket
        
    Returns:
        np.ndarray: The loaded numpy array
        
    Raises:
        Exception: If there's an error loading the numpy file
    """
    s3_path = f"{prefix}/{key}" if prefix else key
    logger.info(f"Loading numpy array from s3://{bucket}/{s3_path}")
    
    try:
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_path)
        return np.load(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        logger.error(f"Error loading numpy array from S3: {str(e)}")
        raise


def load_faiss_from_s3(key: str, bucket: str = S3_BUCKET, prefix: str = S3_PREFIX) -> faiss.Index:
    """Load a FAISS index from S3.
    
    Args:
        key (str): The name of the FAISS index file to load
        bucket (str): The S3 bucket name
        prefix (str, optional): The prefix (folder path) in the bucket
        
    Returns:
        faiss.Index: The loaded FAISS index
        
    Raises:
        Exception: If there's an error loading the FAISS index
    """
    s3_path = f"{prefix}/{key}" if prefix else key
    logger.info(f"Loading FAISS index from s3://{bucket}/{s3_path}")
    
    try:
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_path)
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(obj["Body"].read())
            tmp.flush()
            return faiss.read_index(tmp.name)
    except Exception as e:
        logger.error(f"Error loading FAISS index from S3: {str(e)}")
        raise


def load_bertopic_from_s3(key: str, bucket: str = S3_BUCKET, prefix: str = S3_PREFIX, model_name: str = "all-MiniLM-L6-v2") -> BERTopic:
    """Load a BERTopic model from S3.
    
    Args:
        key (str): The name of the BERTopic model directory (without .zip extension)
        bucket (str): The S3 bucket name
        prefix (str, optional): The prefix (folder path) in the bucket
        model_name (str, optional): The name of the SentenceTransformer model to use
        
    Returns:
        BERTopic: The loaded BERTopic model
        
    Raises:
        Exception: If there's an error loading the BERTopic model
    """
    s3_path = f"{prefix}/{key}.zip" if prefix else f"{key}.zip"
    logger.info(f"Loading BERTopic model from s3://{bucket}/{s3_path}")
    
    try:
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_path)
        zip_bytes = io.BytesIO(obj["Body"].read())

        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, f"{key}.zip")
            
            # Save zip to file
            with open(zip_path, "wb") as f:
                f.write(zip_bytes.getvalue())
            
            # Extract the zip file
            shutil.unpack_archive(zip_path, extract_dir=os.path.join(tmpdir, key), format="zip")
            
            # Load BERTopic model with the specified embedding model
            model_path = os.path.join(tmpdir, key)
            embedding_model = SentenceTransformer(model_name)
            topic_model = BERTopic.load(model_path, embedding_model=embedding_model)
            return topic_model
    except Exception as e:
        logger.error(f"Error loading BERTopic model from S3: {str(e)}")
        raise

@cache
def load_model(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    return load_bertopic_from_s3("bertopic_model_dir", bucket=bucket, prefix=prefix)

@cache
def load_embeddings(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    return load_numpy_from_s3("embeddings_normalized.npy", bucket=bucket, prefix=prefix)

@cache
def load_faiss_index(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    return load_faiss_from_s3("patent_faiss_normalized_embeddings.index", bucket=bucket, prefix=prefix)

@cache
def load_patent_to_idx(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    return load_pickle_from_s3("patent_to_idx.pkl", bucket=bucket, prefix=prefix)

@cache
def load_df(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    return load_pickle_from_s3("df_full.pkl", bucket=bucket, prefix=prefix)

@cache
def load_topics_count_df(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    return load_pickle_from_s3("df_topics_count.pkl", bucket=bucket, prefix=prefix)

@cache
def load_topics_by_year_df(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    return load_pickle_from_s3("df_topics_by_year.pkl", bucket=bucket, prefix=prefix)
