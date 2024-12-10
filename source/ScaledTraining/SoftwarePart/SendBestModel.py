import paho.mqtt.client as paho
from paho import mqtt
import os
import dotenv
from SoftwarePart.SftpHandler import upload_file_to_sftp

def sendBestModel(reward_mean, reward_std):
    dotenvFile = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenvFile, override=True)
    filename = os.getenv("SENDABLMODEL")
    
    try:
        sendMetaData(reward_mean, reward_std, filename)
        SendModelSFTP(filename, dotenvFile)
    except Exception as e:
        print(f"Failed to send best model: {e}")
        raise  


def sendMetaData(reward_mean, reward_std, filename):

    mqttusername = os.getenv('MQTT_USER')
    mqttpassword = os.getenv('MQTT_PASSWORD')
    mqttClient = os.getenv('MQTT_CLIENT')
    mqttPort = int(os.getenv('MQTT_PORT'))
    Username = os.getenv('NvidiaUSERNAME')
    client = paho.Client(client_id="", userdata=None)

    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set(mqttusername, mqttpassword)

    try:
        client.connect(mqttClient, mqttPort)
        client.loop_start()  

        payload = f"TestResultat|{filename}|{reward_mean}|{reward_std}"
        result = client.publish(f"Server/{Username}/Commands", payload=payload, qos=1)
        result.wait_for_publish() 
    finally:
        client.disconnect()
        client.loop_stop()  


def SendModelSFTP(filename, dotenvFile):
    ModelPath = os.path.join("Models")
    ModelPath = os.path.abspath(ModelPath)
    BestModelPath = os.path.join(ModelPath, f"Archive/{filename}")
    print(BestModelPath)
    
    SFTPHost = os.getenv("SFTPHost")
    SFTPPORT = int(os.getenv("SFTP_PORT"))
    SFTPUSERNAME = os.getenv("SFTP_USER")
    SFTPPASSWORD = os.getenv("SFTP_PASSWORD")
    
    try:
        upload_file_to_sftp(SFTPHost, SFTPPORT, SFTPUSERNAME, SFTPPASSWORD, BestModelPath, filename)
    except Exception as e:
        print(f"Error uploading file to SFTP: {e}")
        raise  

    dotenv.set_key(dotenvFile, "SENDABLMODEL", "")
