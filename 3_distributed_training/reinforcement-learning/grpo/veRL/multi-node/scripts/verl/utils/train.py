import time
import argparse
import random
import subprocess  # Added to catch specific subprocess exception
import sagemaker_ssh_helper

def retry_ssh_setup(max_retries=5):
    """Retry the SSH setup function if it fails"""
    for attempt in range(max_retries):
        try:
            print(f"SSH setup attempt {attempt + 1}/{max_retries}")
            
            # Try to set up and start SSH
            sagemaker_ssh_helper.setup_and_start_ssh()
            
            print("SSH setup completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            # This is the specific exception that will be raised if the SSH setup fails
            print(f"Attempt {attempt + 1} failed: Command '{e.cmd}' returned non-zero exit status {e.returncode}")
            print(f"Error output: {e.stderr if hasattr(e, 'stderr') else 'Not available'}")
            
            if attempt < max_retries - 1:
                # Add increasing backoff with some randomness
                backoff_time = (30 * (2 ** attempt)) + random.randint(1, 30)
                print(f"Waiting {backoff_time} seconds before next attempt...")
                time.sleep(backoff_time)
        except Exception as e:
            # Catch any other exceptions
            print(f"Attempt {attempt + 1} failed with unexpected error: {str(e)}")
            
            if attempt < max_retries - 1:
                backoff_time = (30 * (2 ** attempt)) + random.randint(1, 30)
                print(f"Waiting {backoff_time} seconds before next attempt...")
                time.sleep(backoff_time)
    
    print("Failed to set up SSH after all attempts")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_run", type=int, default=86400)
    args, _ = parser.parse_known_args()

    # Try to set up SSH with retries
    ssh_success = retry_ssh_setup(max_retries=5)
    
    # Keep the instance running regardless of SSH setup success
    print(f"Training placeholder job running for {args.max_run} seconds")
    time.sleep(args.max_run)