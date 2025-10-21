import boto3

def get_last_job_name(job_name_prefix):
    sagemaker_client = boto3.client('sagemaker')

    matching_jobs = []
    next_token = None

    while True:
        # Prepare the search parameters
        search_params = {
            'Resource': 'TrainingJob',
            'SearchExpression': {
                'Filters': [
                    {
                        'Name': 'TrainingJobName',
                        'Operator': 'Contains',
                        'Value': job_name_prefix
                    },
                    {
                        'Name': 'TrainingJobStatus',
                        'Operator': 'Equals',
                        'Value': "Completed"
                    }
                ]
            },
            'SortBy': 'CreationTime',
            'SortOrder': 'Descending',
            'MaxResults': 100
        }

        # Add NextToken if we have one
        if next_token:
            search_params['NextToken'] = next_token

        # Make the search request
        search_response = sagemaker_client.search(**search_params)

        # Filter and add matching jobs
        matching_jobs.extend([
            job['TrainingJob']['TrainingJobName'] 
            for job in search_response['Results']
            if job['TrainingJob']['TrainingJobName'].startswith(job_name_prefix)
        ])

        # Check if we have more results to fetch
        next_token = search_response.get('NextToken')
        if not next_token or matching_jobs:  # Stop if we found at least one match or no more results
            break

    if not matching_jobs:
        raise ValueError(f"No completed training jobs found starting with prefix '{job_name_prefix}'")

    return matching_jobs[0]