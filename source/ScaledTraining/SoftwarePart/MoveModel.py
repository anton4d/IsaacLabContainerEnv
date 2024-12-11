import dotenv 
import shutil
from datetime import datetime
import os
import logging




def moveModel(source_path):
    dotenvFile = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenvFile)
    
    ModelPath = os.path.join("Models")
    ModelPath = os.path.abspath(ModelPath)
    ArchiveModelpath = os.path.join(ModelPath,"Archive")
    BestModelPath = os.path.join(ModelPath,"CurrentBestModel")
    username = os.getenv('NvidiaUSERNAME') 


    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')


    newFilename = f"{username}_{current_time}.pt"
    dotenv.set_key(dotenvFile,"SENDABLMODEL",newFilename)


    destinationArchive_path = os.path.join(ArchiveModelpath, newFilename)
    destinationBestModel_Path = os.path.join(BestModelPath, "best_agent.pt")

    try:
        # Move and rename the file
        shutil.copy(source_path, destinationArchive_path)
        logging.info(f"File copied and renamed to: {destinationArchive_path}")

        shutil.copy(source_path, destinationBestModel_Path)
        logging.info(f"File copied and renamed to: {destinationBestModel_Path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")