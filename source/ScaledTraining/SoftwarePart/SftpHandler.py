import paramiko
import os

def upload_file_to_sftp(host, port, username, password, localFilePath, fileName):
    try:
        remoteDir = "uploads"
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print(f"Connecting to SFTP server: {host}:{port} with username: {username}")
        client.connect(hostname=host, port=port, username=username, password=password)
        
        sftp = client.open_sftp()

        try:
            sftp.chdir(remoteDir)
        except IOError:
            print(f"Directory {remoteDir} does not exist.")
            raise Exception(f"Directory {remoteDir} does not exist.")

        if not os.path.isfile(localFilePath):
            print(f"Error: File {fileName} does not exist.")
            raise Exception(f"Error: File {fileName} does not exist.")

        remoteFile = fileName
        print(f"Uploading {fileName} to {remoteDir}")
        sftp.put(localFilePath, remoteFile)
        print(f"File {localFilePath} successfully uploaded to {remoteDir}.")

        sftp.close()
        client.close()
    
    except Exception as e:
        print(f"Failed to upload file to the SFTP server: {e}")
        raise  

def Download_file_from_sftp(host,port,username,password,localFilePath,fileName):
    try:
        remoteDir = "Downloads"
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print(f"Connecting to SFTP server: {host}:{port} with username: {username}")
        client.connect(hostname=host, port=port, username=username, password=password)
        
        sftp = client.open_sftp()

        try:
            sftp.chdir(remoteDir)

        except IOError:
            print(f"Directory {remoteDir} does not exist.")
            return

        print(f"downloading {fileName} to {localFilePath}")
        sftp.get(fileName, localFilePath)
        print(f"File {fileName} successfully downloaded to {localFilePath}.")

        sftp.close()
        client.close()
    except Exception as e:
        print(f"Failed to downloaded file from the SFTP server: {e}")
