import os
import re
import subprocess
import datetime
import numpy as np
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import uuid

bin_dir = "/tmp/bin_executables"
os.makedirs(bin_dir, exist_ok=True)

def _process_test_case(args):
    """Helper function to process a single test case"""
    bin_file, test_input, test_output, prob_name, idx, execution_timeout = args
    run_output = run_code(
        bin_file,
        test_input,
        f"{prob_name}_{idx}",
        execution_timeout=execution_timeout
    )
    
    if run_output is not None:
        result = evaluate_output(test_output, run_output)
        return result["nb_matches"]
    return 0

def run_test_cases_parallel(bin_file: str, 
                           test_inputs: List[str], 
                           test_outputs: List[str], 
                           prob_name: str, 
                           execution_timeout: float,
                           max_test_cases: int = 100,  
                           max_workers: int = 100,     
                           ) -> Tuple[int, int]:
    """
    Run a set of test cases in parallel and return the number of matches and total tests.
    
    Args:
        bin_file: Path to the compiled binary
        test_inputs: Numpy Array of test case inputs
        test_outputs: Numpy Array of expected outputs
        prob_name: Name of the problem (for logging)
        execution_timeout: Timeout for each test case
        max_test_cases: Maximum number of test cases to run (limits sample size)
        max_workers: Maximum number of parallel workers (default: 32, ~2/3 of physical cores)
    
    Returns:
        Tuple of (total_matches, total_tests)
    """
    if len(test_inputs) != len(test_outputs):
        total_available_tests = min(len(test_inputs), len(test_outputs))
    else:
        total_available_tests = len(test_inputs)
    
    if total_available_tests > max_test_cases:
        random_indices = np.random.choice(total_available_tests, size=max_test_cases, replace=False)        
        test_inputs = test_inputs[random_indices]
        test_outputs = test_outputs[random_indices]
        
    total_tests = len(test_inputs)
    
    effective_workers = min(max_workers, total_tests)
    
    args_list = [
        (bin_file, test_input, test_output, prob_name, i, execution_timeout) 
        for i, (test_input, test_output) in enumerate(zip(test_inputs, test_outputs))
    ]
    
    total_matches = 0
    try:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            chunk_size = max(1, len(args_list) // (2 * effective_workers))            
            results = list(executor.map(_process_test_case, args_list, chunksize=chunk_size))
            total_matches = sum(results)
            
    except Exception as e:
        pass
    
    msg = f"{total_matches}/{total_tests} tests passed with ({effective_workers} workers) - {prob_name}"
    print(msg)
            
    return total_matches, total_tests

def extract_solution(solution_str, num_cpp=None):
    """Extract the code from the solution string."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    code_pattern = r'<code>(.*?)</code>'
    matches = list(re.finditer(code_pattern, solution_str, re.DOTALL))
    if matches:
        code_content = matches[-1].group(1).strip()
        cpp_pattern = r'```cpp\n(.*?)```'
        cpp_matches = list(re.finditer(cpp_pattern, code_content, re.DOTALL))
        if cpp_matches:
            final_answer = cpp_matches[-1].group(1).strip()
            if num_cpp is not None:
                num_cpp.append(len(cpp_matches))
                
        else:
            final_answer = code_content
    else:
        cpp_pattern = r'```cpp\n(.*?)```'
        matches = list(re.finditer(cpp_pattern, solution_str, re.DOTALL))
        if matches:
            final_answer = matches[-1].group(1).strip()
            if num_cpp is not None:
                num_cpp.append(len(matches))
        else:
            final_answer = None

    return final_answer

def compile_code(code_str, prob_name):
    short_uuid = str(uuid.uuid4())[:8]
    code_name = f"{prob_name}_{short_uuid}.cpp"
    code_path = Path(bin_dir) / code_name
    code_path.write_text(code_str)
    bin_file = code_path.with_suffix(".bin")
    compile_command = f'g++ -std=c++17 -O2 "{code_path}" -o "{bin_file}"'
    try:
        _ = subprocess.run(
            compile_command, 
            shell=True, 
            check=True,
            capture_output=True, 
            text=True,  
            encoding='utf-8',    
            errors='replace'    
        )
    except subprocess.CalledProcessError as cpe:
        #err_msg = "\n".join(cpe.stderr.split("\n")[:2])
        #print(f"Compiling {prob_name} failed: {cpe.returncode=}, {err_msg=}")
        return None
    return bin_file


def evaluate_output(expected: str, actual: str) -> dict:
    """
    Check a single test case output against expected output.
    Each output may contain multiple lines that need to match exactly.

    Example expected format:
    "20\n3 6 7 4 5 \n"
    """
    if actual is None:
        return {"matches": False, "nb_matches": 0, "total": 1}

    expected = expected.strip()
    actual = actual.strip()

    matches = expected == actual

    return {"matches": matches, "nb_matches": 1 if matches else 0, "total": 1}


def run_code(bin_file, input_str, prob_name, execution_timeout=5):
    """Using fixed timeout for training efficiency"""
    try:
        result = subprocess.run(
            bin_file,
            input=input_str,
            capture_output=True,
            text=True,
            check=True,
            timeout=execution_timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        #print(f"{prob_name} execution timeout after {execution_timeout} seconds")
        return None
    except subprocess.CalledProcessError as cpe:
        #print(f"{prob_name} execution failed: {cpe.returncode=}, {cpe.stderr=}")
        return None
    except Exception as e:
        #print(e)
        return None
    return result.stdout.strip()


def to_valid_path(string):
    """
    Convert a string to a valid file path by removing/replacing invalid characters
    and replacing spaces with hyphens.
    """
    invalid_chars = '<>:"/\\|?*'
    replacement_char = "_"

    valid_string = "".join(
        replacement_char if c in invalid_chars else c for c in string
    )

    valid_string = valid_string.replace(" ", "-")

    valid_string = valid_string.strip().strip(".")

    while "__" in valid_string:
        valid_string = valid_string.replace("__", "_")
    while "--" in valid_string:
        valid_string = valid_string.replace("--", "-")

    return valid_string


def run_test_cases(
    bin_file: str,
    test_inputs: List[str],
    test_outputs: List[str],
    prob_name: str,
    execution_timeout: float = 1.5,
) -> Tuple[int, int]:
    """
    Run a set of test cases and return the number of matches and total tests.

    Args:
        bin_file: Path to the compiled binary
        test_inputs: List of test case inputs
        test_outputs: List of expected outputs
        prob_name: Name of the problem (for logging)
        execution_timeout: Timeout for each test case

    Returns:
        Tuple of (total_matches, total_tests)
    """
    total_matches = 0
    total_tests = len(test_inputs)

    for test_input, test_output in zip(test_inputs, test_outputs):
        run_output = run_code(
            bin_file, test_input, prob_name, execution_timeout=execution_timeout
        )

        if run_output is not None:
            result = evaluate_output(test_output, run_output)
            total_matches += result["nb_matches"]

    return total_matches, total_tests


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    """Main scoring function for code_contests dataset problems"""
    nb_cpps = []
    code = extract_solution(solution_str=solution_str, num_cpp=nb_cpps)
    bin_file = None

    # Create problem name from code_contests dataset
    if "name" in ground_truth:
        gt_prob_name = ground_truth["name"]
    elif (
        "competition_id" in ground_truth and "problem_id" in ground_truth
    ):
        gt_prob_name = f'{ground_truth["competition_id"]}-{ground_truth["problem_id"]}'
    else:
        raise Exception("Problem no name!")
    prob_name = f"code_contests_{to_valid_path(gt_prob_name)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        if code is None:
            return -1

        bin_file = compile_code(code, prob_name)
        if not bin_file:
            return -0.5
    
        #sampled_inputs = ground_truth["public_tests"]["input"]
        #sampled_outputs = ground_truth["public_tests"]["output"]
        private_inputs = ground_truth["private_tests"]["input"]
        private_outputs = ground_truth["private_tests"]["output"]

        total_matches, total_tests = run_test_cases_parallel(
            bin_file=bin_file,
            test_inputs=private_inputs,
            test_outputs=private_outputs,
            prob_name=f"{prob_name}_private",
            execution_timeout=1.5, #TODO - configure item
            max_test_cases=1000, 
            max_workers=100,  
        )

        private_score = total_matches / total_tests if total_tests > 0 else 0
        if private_score < 1:
            private_score = 0
        if private_score > 1:
            private_score = 1
        return private_score

    finally:
        if bin_file:
            try:
                if os.path.exists(bin_file):
                    os.remove(bin_file)
            except Exception as e:
                pass