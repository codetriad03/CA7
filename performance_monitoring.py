import subprocess

def execute_mlflow_script():
    subprocess.run(['mlflow', 'run', 'https://github.com/codetriad03/CA7.git', '-P', 'entry_point=main'])
    # Add code to monitor performance metrics and compare with thresholds

# Execute MLflow project script
execute_mlflow_script()
