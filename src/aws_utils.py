from pathlib import Path
import logging
import boto3

logger = logging.getLogger(__name__)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)

def upload_artifacts(artifacts: Path, config: dict) -> list[str]:
    """Upload all the artifacts in the specified directory to S3


    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3; see example config file for structure

    Returns:
        List of S3 uri's for each file that was uploaded
    """
    session = boto3.Session()
    s3 = session.client('s3')

    try:
        # List of uploaded file paths
        uploaded_files = []
        bucket_name = config['aws']['bucket_name']

        artifacts_path = Path(artifacts)

        # Iterate over files in the directory
        for file_path in artifacts_path.glob('*'):
            if file_path.is_file():
                # Construct S3 key (object key)
                s3_key = str(file_path.name)

                # Upload file to S3
                print('KEY NAME:',s3_key)
                s3.upload_file(str(file_path), bucket_name, s3_key)

                # Append S3 URI to the list
                uploaded_files.append(f"s3://{bucket_name}/{s3_key}")

        logger.info('Artifacts successfully uploaded to s3!')
        return uploaded_files

    except NameError as e:
        logger.error('Artifacts could not be uploaded to s3: %s', e)
