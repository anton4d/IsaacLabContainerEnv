import time
import paho.mqtt.client as paho
from paho import mqtt
import dotenv 
import os
import subprocess
import shutil
import sys
from SoftwarePart.SftpHandler import Download_file_from_sftp

training_process = None
Username = None
def on_connect(client, userdata, flags, rc):
    print("Connect received with code %s." % rc)


def on_publish(client, userdata, mid):
    print("mid: " + str(mid))

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_message(client, userdata, msg):
    
    message = msg.payload.decode('utf-8')
    print(f"Received message on topic {msg.topic}: {message}")
    handle_message(client, message)

def handle_message(client, message):
    global Username
    try:
        Command, data = message.split("|",1)
        handler = command_dispatcher.get(Command, handle_unknown)
        if handler != handle_unknown:
            handler(client, Username, data)
        else:
            handler(client, Username, Command)
    except ValueError:
        print(f"Malformed message: {message}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def handle_stopTrain(client,Username, message):
    global training_process
    if training_process:
        print(f"Stopping the training loop with data: {message}")
        training_process.terminate()
        training_process.wait()  # Ensure the process is fully terminated
        training_process = None
        print("Training loop stopped.")
        payload = f"LoopStoped|"
        client.publish(f"Server/{Username}/Commands/Status", payload=payload, qos=1)
    else:
        print("No training process is running.")
        payload = f"LoopStope|"
        client.publish(f"Server/{Username}/Commands/Status", payload=payload, qos=1)


def StartTrain(client, Username, message):
    global training_process
    if training_process and training_process.poll() is None:
        print("Training process is already running.")
        return

    try:
        print(f"Starting the training loop with data: {message}")
        LoopPath = os.path.abspath("Loop.py")

        if not os.path.isfile(LoopPath):
            raise FileNotFoundError(f"Loop.py not found at {LoopPath}")

        # Start the training process without logging to a file
        training_process = subprocess.Popen(
            [sys.executable, LoopPath]
        )

        if training_process.poll() is None:
            print(f"Training loop started successfully.")
            payload = "LoopStarted|"
            client.publish(f"Server/{Username}/Commands/Status", payload=payload, qos=1)
        else:
            print("Failed to start the training loop.")
            payload = "LoopError|Failed to start the training loop."
            client.publish(f"Server/{Username}/Commands/Error", payload=payload, qos=1)

    except FileNotFoundError as fnf_error:
        print(f"File error: {fnf_error}")
        payload = f"FileError|File error: {fnf_error}"
        client.publish(f"Server/{Username}/Commands/Error", payload=payload, qos=1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        payload = f"UxcepError|An unexpected error occurred: {e}"
        client.publish(f"Server/{Username}/Commands/Error", payload=payload, qos=1)



def handle_startTrain(client,Username, message):
    StartTrain(client,Username,message)


def DownloadFile(fileName):
    ModelPath = os.path.join("Models")
    ModelPath = os.path.abspath(ModelPath)
    if not os.path.exists(ModelPath):
        os.makedirs(ModelPath)
        print(f"Directory created: {ModelPath}")
    NewModelDir = os.path.join("NewModel")
    if not os.path.exists(NewModelDir):
        os.makedirs(NewModelDir)
        print(f"Directory created: {NewModelDir}")
    BestModelPath = os.path.join(ModelPath,NewModelDir,"best_agent.pt")
    SFTPHost = os.getenv("SFTPHost")
    SFTPPORT = int(os.getenv("SFTP_PORT"))
    SFTPUSERNAME = os.getenv("SFTP_USER")
    SFTPPASSWORD = os.getenv("SFTP_PASSWORD")
    print(f"try to download file{fileName}")
    Download_file_from_sftp(SFTPHost,SFTPPORT, SFTPUSERNAME, SFTPPASSWORD, BestModelPath, fileName)

def handle_newSetupData(client,Username, message):
    dotenvFile = dotenv.find_dotenv()
    try:
        maxIteration, seed, testIteration, testSeed, fileName = message.split("|",4)
        dotenv.set_key(dotenvFile, "MaxITERATION", maxIteration)
        dotenv.set_key(dotenvFile, "SEED", seed)
        dotenv.set_key(dotenvFile, "TESTITERATION", testIteration)
        dotenv.set_key(dotenvFile, "TESTSEED", testSeed)
        dotenv.set_key(dotenvFile, "NewModel", "True")  
        DownloadFile(fileName)
    except ValueError:
        print(f"Malformed data in message data: {message}")
        payload = f"SetupError|Malformed data in message data: {message}"
        client.publish(f"Server/{Username}/Commands/Error", payload=payload, qos=1)
    finally:
        payload = f"SetupDataSuccesfull|"
        client.publish(f"Server/{Username}/Commands/Status", payload=payload, qos=1)

    print(f"The env will get new data with the data {message}")

def handle_newModel(client, Username, message):
    print(f"The training loop has been given a new model with data {message}")
    dotenvFile = dotenv.find_dotenv()
    dotenv.set_key(dotenvFile, "NewModel", "True")
    DownloadFile(message)  


def handle_setup(client,Username, message):
    dotenvFile = dotenv.find_dotenv()
    try:
        maxIteration, seed, testIteration, testSeed, fileName = message.split("|",4)
        dotenv.set_key(dotenvFile, "MaxITERATION", maxIteration)
        dotenv.set_key(dotenvFile, "SEED", seed)
        dotenv.set_key(dotenvFile, "TESTITERATION", testIteration)
        dotenv.set_key(dotenvFile, "TESTSEED", testSeed)
        dotenv.set_key(dotenvFile, "NewModel", "True")  
        LoopPath = os.path.join("Loop.py")
        LoopPath = os.path.abspath(LoopPath)
        print(LoopPath)

        DownloadFile(fileName)
        StartTrain(client,Username, message)
    except ValueError:
        print(f"Malformed data in message data: {message}")
        payload = f"LoopError|Malformed data in message data: {message}"
        client.publish(f"Server/{Username}/Commands/Error", payload=payload, qos=1)

    print(f"The training loop will be setup and started with data {message}")


def handle_unknown(client,Username, command):
        print(f"Unknown command: {command}")
        payload = f"Unknown|{command}"
        client.publish(f"Server/{Username}/Commands/Error", payload=payload, qos=1)

command_dispatcher = {
            "StopTrain": handle_stopTrain,
            "StartTrain": handle_startTrain,
            "NewSetupData": handle_newSetupData,
            "NewModel": handle_newModel,
            "Setup": handle_setup
            # Add more commands here if necessary
        }

def main():
    global Username
    dotenvFile = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenvFile)

    mqttusername = os.getenv('MQTT_USER')
    mqttpassword = os.getenv('MQTT_PASSWORD')
    mqttClient = os.getenv('MQTT_CLIENT')
    mqttPort = int(os.getenv('MQTT_PORT'))
    Username = os.getenv('NvidiaUSERNAME')
    client = paho.Client(client_id="", userdata=None)
    client.on_connect = on_connect


    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set(mqttusername,mqttpassword)
    client.connect(mqttClient, mqttPort)

    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.on_publish = on_publish


    client.subscribe(f"{Username}/Commands", qos=1)
    client.subscribe('all/Commands', qos=1)

    payload = f"NewUser|{Username}"
    client.publish("Server/Commands", payload=payload, qos=1)

    client.loop_forever()

if __name__ == "__main__":
    main()