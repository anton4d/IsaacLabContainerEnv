import os
import subprocess
import logging
import time
import sys
import dotenv


def setup_logging():
    log_filename = "loop.log"  
    log_filepath = os.path.join("logs", log_filename)
    

    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    logging.basicConfig(filename=log_filepath, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    return log_filepath

def call_train(dotenvFile):

    Seed = os.getenv('SEED')
    maxIterations = os.getenv('MaxITERATION')
    ENVS = os.getenv('ENVS')

    newModel = os.getenv('NewModel')
    
    if not Seed or not maxIterations:
        print("Error: Seed and MaxIterations environment variables must be set.")
        logging.error("Error: Seed and MaxIterations environment variables must be set.")
        return
    
    try:
        models_path = os.path.abspath("Models")
        if newModel == "True":
            modelPath = os.path.join(models_path, "NewModel/best_agent.pt")
            dotenv.set_key(dotenvFile, "NewModel", "False")
        else:
            modelPath = os.path.join(models_path, "CurrentBestModel/best_agent.pt")
        
        SKRL_train_path = os.path.abspath("RobotPart/SKRL_train.py")
        
        logging.info(f"Starting the training subprocess with model: {modelPath} and envs, Seed {Seed}, envs {ENVS}, MaxIteration {maxIterations}")
        
        subprocess.run([
            sys.executable, SKRL_train_path,
            '--headless',
            f'--seed={Seed}',
            f'--max_iterations={maxIterations}',
            f'--load_model={modelPath}',
            f'--num_envs={ENVS}'
        ], check=True)
        
        logging.info("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred in SKRL_train.py: {e}")

def call_test():
    dotenvFile = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenvFile)
    Seed = os.getenv('TESTSEED')
    maxIterations = os.getenv('TESTITERATION')
    ENVS = os.getenv('ENVS')
    
    if not Seed or not maxIterations:
        print("Error: Seed and MaxIterations environment variables must be set.")
        return
    
    try:
        test_performance_path = os.path.abspath("RobotPart/test_performance.py")
        
        logging.info(f"Starting the test performance subprocess with envs, Seed {Seed}, envs {ENVS}, MaxIteration {maxIterations}")
        
        subprocess.run([
            sys.executable, test_performance_path,
            '--headless',
            f'--seed={Seed}',
            f'--max_iterations={maxIterations}',
            f'--num_envs={ENVS}'
        ], check=True)
        
        logging.info("Test completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred in test_performance.py: {e}")

def call_optimized_envs():
    try:
        optimizedEnvs_path = os.path.abspath("RobotPart/SKRL_calculate_optimized_envs.py")
        
        logging.info("Starting optimized environments calculation.")
        
        subprocess.run([
            sys.executable, optimizedEnvs_path, "--headless"
        ], check=True)
        
        logging.info("Optimized environments calculation completed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred in SKRL_calculate_optimized_envs.py: {e}")

def main():

    log_filepath = setup_logging()
    logging.info("Starting main loop")

    call_optimized_envs()

    while True:
        dotenvFile = dotenv.find_dotenv()
        dotenv.load_dotenv(dotenvFile, override=True)
        
        logging.info("Calling training process.")
        call_train(dotenvFile)
        
        logging.info("Calling test process.")
        call_test()
        
        print("Cycle complete. Restarting...")
        logging.info("Cycle complete. Restarting...")
        time.sleep(5)

if __name__ == "__main__":
    main()
