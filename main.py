# Parsing imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.pipeline_options import PictureDescriptionApiOptions, PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling_core.types.doc.document import PictureItem
import pymupdf

# ML and functionalities import
import torch, gc
import ctypes
from sys import argv
import json
from pprint import pprint
import subprocess, threading
from uuid import uuid4

# Extra functionalities import
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from datetime import timezone, timedelta
from datetime import datetime
import requests

# Azure dependencies
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient

# AWS dependencies
import boto3
import botocore

# Chunking and embedding dependencies
# from markdown_splitter import MarkdownSplitter
from latest_embed import start_embed, orchestrator

load_dotenv()

# ANSI escape codes
RESET = "\033[0m"
COLORS = {
'DEBUG': "\033[36m", # Cyan
'INFO': "\033[32m", # Green
'WARNING': "\033[33m", # Yellow
'ERROR': "\033[31m", # Red
'CRITICAL': "\033[41m", # Red background
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt="%d-%B-%Y %H:%M:%S",
    filemode="w",
    filename=str(Path(__file__).parent)+"/parser.log"
)

logging.getLogger("azure.cosmos").setLevel(logging.WARNING)

lambda_client=boto3.client("lambda", region_name="ap-south-1")

def get_instance_id():
    TOKEN_URL = "http://169.254.169.254/latest/api/token"
    META_URL = "http://169.254.169.254/latest/meta-data/instance-id"

    token = requests.put(
        TOKEN_URL,
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        timeout=2
    ).text

    instance_id = requests.get(
        META_URL,
        headers={"X-aws-ec2-metadata-token": token},
        timeout=2
    ).text

    return instance_id

def put_event(instance_id:str, job_id:str)-> bool:
    try:
        response = lambda_client.invoke(
            FunctionName="UpdateMachineEvent",
            InvocationType="RequestResponse",  # or "Event" for async
            Payload=json.dumps({
                "instance_id": instance_id,
                "job_id": job_id
            })
        )
        return True
    except Exception as e:
        print(e)
        return False
    
ec2 = boto3.client("ec2", region_name="ap-south-1")
def create_instance_tag(instance_id: str, key: str, value: str):
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[
            {
                "Key": key,
                "Value": value
            }
        ]
    )

    print(f"Tag '{key}' set to '{value}' on {instance_id}")
    
def ping(instance_id:str, job_id:str, stop_event):
    print("Ping thread started")
    while not stop_event.is_set():
        try:
            lambda_client.invoke(
                FunctionName="UpdateMachineEvent",
                InvocationType="Event",  # or "Event" for async
                Payload=json.dumps({
                    "instance_id": instance_id,
                    "job_id": job_id
                })
            )
            print("Pinged")

            stop_event.wait(30)
            # return True
        except Exception as e:
            print(e)
            # return False
    else:
        print("Ping stopped")

def time_in_ist()->str:
    gmt_plus_530 = timezone(timedelta(hours=5, minutes=30))
    current_time = datetime.now(gmt_plus_530)
    formatted_time=current_time.strftime("%Y-%m-%dT%H:%M:%S")
    logging.info(f"Current time (GMT+05:30): {formatted_time}")
    return str(formatted_time)

def elapsed_time(start_time, end_time):
    elapsed=datetime.fromisoformat(end_time)-datetime.fromisoformat(start_time)
    hours, remainder=divmod(elapsed.total_seconds(), 3600)
    minutes, seconds=divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

class Cosmos:
    def __init__(self):
        self.subscription_id="37d0150c-636f-4d07-9065-dcc304016fa2"
        self.resource_group="digihumans-instances"
        self.credential=DefaultAzureCredential()
        
        self.cosmos_url=os.getenv("COSMOS_URL")
        self.key=os.getenv("COSMOS_KEY")
        self.db_name="digihumansData"
        self.container="parserLogs"
        self.embedding_container="embeddingLogs"

        self.client=CosmosClient(url=self.cosmos_url,credential=self.key, logging_enable=False)

    def populate_container(self,items:dict, embedding:bool=False):
        db=self.client.get_database_client(self.db_name)
        if embedding:
            con=db.get_container_client(self.embedding_container)
        else:
            con=db.get_container_client(self.container)
        try:
            con.create_item(body=items)
            logging.info("Data uploaded")
        except exceptions.CosmosResourceExistsError:
            logging.info("Cosmos error: Resource already exists")
            # if items.get("id")=="sqs":
                # self.delete_entry_and_update(items=items)
        except exceptions.CosmosResourceNotFoundError:
            logging.info("Entity with the specified id does not exist in the system.")
        except Exception as e:
            print("Error: ",e)

    def delete_entry_and_update(self, items:dict, embedding:bool=False):
        database=self.client.get_database_client(self.db_name)
        if embedding:
            con=database.get_container_client(self.embedding_container)
        else:
            con=database.get_container_client(self.container)
        id=items.get("id")
        user_id=items.get("user_id")
        try:
            con.delete_item(item=id, partition_key=user_id)
            logging.info(f"Item with item id {id} is deleted from cosmos DB")
            self.populate_container(items, embedding)
        except exceptions.CosmosResourceNotFoundError as resource_not_found_error:
            logging.info(f"Resource not found for 'id': {id}, 'user_id': {user_id}")

class SQS:
    def __init__(self):
        self.client=boto3.client('sqs', region_name='ap-south-1')
        self.dev_queue_url="https://sqs.ap-south-1.amazonaws.com/937545714257/document-queue-dev.fifo"
        self.prod_queue_url="https://sqs.ap-south-1.amazonaws.com/937545714257/DocumentParsingJobs.fifo"

        self.end_queue_url="https://sqs.ap-south-1.amazonaws.com/937545714257/DeleteFinishedParsingJobs.fifo"
        self.job_failed_queue_url="https://sqs.ap-south-1.amazonaws.com/937545714257/FailedDocumentsParsingJobs.fifo"

    def send_success_to_sqs(self, body):
        try:
            response=self.client.send_message(
                QueueUrl=self.end_queue_url,
                MessageBody=json.dumps(body),
                MessageGroupId="1"
            )
        except Exception as e:
            print(e)

    def send_failure_to_sqs(self,body):
        try:
            response=self.client.send_message(
                QueueUrl=self.job_failed_queue_url,
                MessageBody=json.dumps(body),
                MessageGroupId="1"
            )
        except Exception as e:
            print(e)

class S3:
    def __init__(self):
        self.s3 = boto3.client('s3', region_name="ap-south-1")

    def download_document_from_S3(self, id:str, user_id:str, object_key:str):
        try:
            bucket_name = "digihumans-user-uploads-for-parsing-prod"
            # object_key = object_key
            download_path = f"{str(Path(__file__).parent)}/raw_data/{id}/"

            os.makedirs(download_path, exist_ok=True)
            file_name=os.path.basename(object_key)
            local_path=os.path.join(download_path, file_name)
            # print(object_key)
            self.s3.download_file(bucket_name, object_key, local_path)
            return True, local_path
        except Exception as e:
            return False, str(e)

class Azure:
    def __init__(self):
        self.connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name="digihumans"

    def download_document_from_Azure(self, id:str, user_id:str):
        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            container_client = blob_service_client.get_container_client(self.container_name)

            prefix = f"client_data/{user_id}/documents/{id}/"

            blobs = container_client.list_blobs(name_starts_with=prefix)

            for blob in blobs:
                if blob.name.endswith("md"):
                    continue
                blob_client = blob_service_client.get_blob_client(container=self.container_name, blob=blob.name)
                download_stream = blob_client.download_blob()
                data = download_stream.readall()
                download_path = f"{str(Path(__file__).parent)}/raw_data/{id}"
                os.makedirs(download_path, exist_ok=True)

                file_name = os.path.basename(blob.name)
                file_path = f"{download_path}/{file_name}"
                with open(file_path,"wb") as f:
                    f.write(data)
                logging.info(f"Blob downloaded and saved to {file_path}")
                return True, file_path
        except Exception as e:
            logging.info(f"Azure error: {str(e)}")
            return False, str(e)
    
    def upload(self, file_path:str, file_name:str, id:str, user_id:str) -> str:
        try:
            # reuse your variables: connection_string, container_name, blob_name, job_id
            # file_path = f"/app/parserv1/data/{job_id}.md"

            blob_name=f"client_data/{user_id}/documents/{id}/{file_name}"

            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            blob_client = blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)

            with open(file_path, "rb") as f:
                blob_client.upload_blob(
                    f,
                    overwrite=True,  # set False if you want it to fail when blob exists
                    # content_settings=ContentSettings(content_type="application/zip")
                )
            msg=f"Uploaded {file_path} to '{blob_name}'"
            logging.info(msg)
            return True, msg
        except Exception as e:
            return False, f"Upload failed\nError: {e}"

class AWS:
    def __init__(self):
        self.BUCKET_NAME="digihumans-llmknowledgebase"
        self.REGION="ap-south-1"
        self.s3=boto3.client("s3", region_name=self.REGION)

    def upload(self, file_path:str, file_name:str, id:str, user_id:str) -> str:
        try:
            blob_name=f"client_data/{user_id}/documents/{id}/{file_name}"
            s3_output=self.s3.upload_file(
                Filename=file_path,
                Bucket=self.BUCKET_NAME,
                Key=blob_name
            )
            msg=f"Uploaded {file_path} to '{blob_name}'"
            return True, msg

        except Exception as e:
            return False, f"Upload failed\nError: {e}"

class Document():
    def __init__(self):
        self.cosmos=Cosmos()
        self.azure=Azure()
        self.aws=AWS()
        self.sqs=SQS()
        self.logs={
            "id":None,
            "user_id":None,
            "doc":None,
            "task":{
                "error":None,
                "msg":None
            },
            "time":{
                "start":"00:00:00",
                "elapsed":"00:00:00",
                "end":"00:00:00",
                "tz":"Asia/Kolkata"
            }
        }

        self.file_path=f"{str(Path(__file__).parent)}/files"
        self.raw_data_file_path=f"{str(Path(__file__).parent)}/raw_data"
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            logging.info(f"{COLORS.get('INFO', 'RESET')}File created{RESET}")
        
        pipeline_options=PdfPipelineOptions(
            # artifacts_path="/home/ubuntu/.cache/docling/models",
            do_picture_description=True,
            do_ocr=True,
            # generate_picture_images=True,
            enable_remote_services=True,
            picture_description_options=PictureDescriptionApiOptions(
                url="http://localhost:11434/v1/chat/completions",
                params={
                    "model":"ministral-3:3b"
                },
                prompt="Describe the image in rich visual detail, including objects, people, environment, colors, layout, actions, and context.\n"
                "If visible text appears anywhere in the image, extract it using OCR and return it in markdown enclosed inside parentheses.\n"
                "If no readable text is present, simply omit the OCR section and continue describing the image normally. Do not mention the absence of text.",
                timeout=600,
                concurrency=1
            ),
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.CUDA
            ),
            ocr_options=EasyOcrOptions()
        )

        self.converter=DocumentConverter(
            format_options={
                InputFormat.PDF:PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    
    def convert_document_to_markdown(self, source:str, id:str, user_id:str, orig_source:str|None=None, dev:bool=False):
        self._startTime=time_in_ist()
        self.logs["time"]["start"]=self._startTime
        self.logs["id"]=id
        self.logs["user_id"]=user_id
        self.logs["doc"]=source.split("/")[-1] if not orig_source else orig_source
        self.cosmos.populate_container(items=self.logs)
        doc=pymupdf.open(filename=source)
        if doc.is_encrypted:
            self.logs["task"]["error"]=True
            self.logs["task"]["msg"]=f"{source.split("/")[-1].split(".")[-1].upper()} is password protected."
            _endTime=time_in_ist()
            
            self.logs["time"]["elapsed"]=elapsed_time(self._startTime, _endTime)
            self.logs["time"]["end"]=_endTime
            self.cosmos.delete_entry_and_update(items=self.logs)
            self.logs={
                "id":None,
                "user_id":None,
                "doc":None,
                "task":{
                    "error":None,
                    "msg":None
                },
                "time":{
                    "start":"00:00:00",
                    "elapsed":"00:00:00",
                    "end":"00:00:00",
                    "tz":"Asia/Kolkata"
                }
            }
            doc.close()

            if not dev:
                # API for calling the lambda with status update = False
                # response=requests.post("https://api.digihumans.ai/app/doc-data-extraction/update-job-status",json={"user-id":user_id,"doc-id":id, "stage":"parsing", "jobSuccessful":False})
                response=requests.post("https://api.digihumans.ai/app/knowledgebase/job-status-update",json={"type":"document", "id":id, "user-id":user_id, "stage":"parsing", "jobSuccessful":False})
                logging.info(f"Response from the API: {response.text}")
                self.sqs.send_failure_to_sqs({"job_id":id})
            return "Password protected Document"
        else:
            doc.close()
        try:
            result=self.converter.convert(source=source)
            document=result.document
            markdown=document.export_to_markdown()
            return self.save(source, markdown, id, user_id)
        except Exception as e:
            self.logs["task"]["error"]=True
            self.logs["task"]["msg"]=f"Document error: {str(e)}"
            _endTime=time_in_ist()

            self.logs["time"]["elapsed"]=elapsed_time(self._startTime, _endTime)
            self.logs["time"]["end"]=_endTime
            self.cosmos.delete_entry_and_update(items=self.logs)

            if not dev:
                # API for calling the lambda with status update = False
                # response=requests.post("https://api.digihumans.ai/app/doc-data-extraction/update-job-status",json={"user-id":user_id,"doc-id":id, "stage":"parsing" ,"jobSuccessful":False})
                response=requests.post("https://api.digihumans.ai/app/knowledgebase/job-status-update",json={"type":"document", "id":id, "user-id":user_id, "stage":"parsing", "jobSuccessful":False})
                logging.info(f"Response from the API: {response.text}")
                self.sqs.send_failure_to_sqs({"job_id":id})

            self.logs={
                "id":None,
                "user_id":None,
                "doc":None,
                "task":{
                    "error":None,
                    "msg":None
                },
                "time":{
                    "start":"00:00:00",
                    "elapsed":"00:00:00",
                    "end":"00:00:00",
                    "tz":"Asia/Kolkata"
                }
            }
            return f"Processing error: {e}"
        finally:
            subprocess.run(["rm", "-r", f"{self.raw_data_file_path}/{id}"])
            gc.collect()
            torch.cuda.empty_cache()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    
    def save(self, file_name:str, markdown:str, id:str, user_id:str, dev:bool=False):
        file_name=file_name.split("/")[-1].replace(file_name.split(".")[-1], "md")
        folder_path=f"{self.file_path}/{id}"
        os.makedirs(folder_path, exist_ok=True)
        file_path=f"{folder_path}/{file_name}"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        output=f"File saved to '{file_path}'"
        # status, msg = self.azure.upload(file_path=file_path, file_name=file_name, id=id, user_id=user_id)
        status, msg = self.aws.upload(file_path=file_path, file_name=file_name, id=id, user_id=user_id)

        if status:
            self.logs["task"]["error"]=False
            self.logs["task"]["msg"]="success"
            api_status=True
        else:
            self.logs["task"]["error"]=True
            self.logs["task"]["msg"]=f"Blob error: {msg}"
            api_status=False

        _endTime=time_in_ist()
        self.logs["time"]["elapsed"]=elapsed_time(self._startTime, _endTime)
        self.logs["time"]["end"]=_endTime

        self.cosmos.delete_entry_and_update(items=self.logs)

        subprocess.run(["rm", "-r", f"{folder_path}"])
        
        if not dev:
            # API for calling the lambda with status update = False
            if api_status:
                # response=requests.post("https://api.digihumans.ai/app/doc-data-extraction/update-job-status",json={"user-id":user_id,"doc-id":id, "stage":"parsing" ,"jobSuccessful":True})
                response=requests.post("https://api.digihumans.ai/app/knowledgebase/job-status-update",json={"type":"document", "id":id, "user-id":user_id, "stage":"parsing", "jobSuccessful":True})
                logging.info(f"Response from save API: {response.text}")
                self.sqs.send_success_to_sqs({"job_id":id})
            else:
                # response=requests.post("https://api.digihumans.ai/app/doc-data-extraction/update-job-status",json={"user-id":user_id,"doc-id":id, "stage":"parsing","jobSuccessful":False})
                response=requests.post("https://api.digihumans.ai/app/knowledgebase/job-status-update",json={"type":"document", "id":id, "user-id":user_id, "stage":"parsing", "jobSuccessful":False})
                
                logging.info(f"Response from save API: {response.text}")
                self.sqs.send_failure_to_sqs({"job_id":id})

        # doc=self.logs.get("doc")
        # self.cosmos.populate_container(items={"id":self.logs.get("id"), "user_id":self.logs.get("user_id"), "doc":doc.replace(doc.split(".")[-1], "md")}, embedding=True)
        logging.info(output)
        self.logs={
            "id":None,
            "user_id":None,
            "doc":None,
            "task":{
                "error":None,
                "msg":None
            },
            "time":{
                "start":"00:00:00",
                "elapsed":"00:00:00",
                "end":"00:00:00",
                "tz":"Asia/Kolkata"
            }
        }
        return markdown

if __name__=="__main__":
    document=Document()
    cosmos=Cosmos()
    sqs=SQS()

    # markdownSplitter=MarkdownSplitter()

    instance_id = get_instance_id()

    queue_dev=sqs.dev_queue_url
    queue_prod=sqs.prod_queue_url

    parent_path=str(Path(__file__).parent)
    logging.info(f"Parent path: {parent_path}")

    stop=False
    count=0

    occupied=False

    dev=False if len(argv[1:])<1 else bool(argv[1])

    if dev:
        azure=Azure()
    else:
        s3=S3()

    while not stop:
        if not os.path.exists("/home/ubuntu/running_check/RUNNING"):
            print("Code needed to be stopped")
            create_instance_tag(instance_id, "status", "inactive")
            stop=True
            occupied=False
            continue
        try:
            response=sqs.client.receive_message(
                QueueUrl=queue_prod if not dev else queue_dev,
                MaxNumberOfMessages=1,        # up to 10 messages
                WaitTimeSeconds=10,            # long polling
                VisibilityTimeout=90           # seconds
            )
        except Exception as e:
            # Log the data in cosmos DB for sqs errors and stop the program
            stop=True
            cosmos.populate_container({"id":"sqs", "user_id":"arsh", "task":{"error":True, "msg":f"SQS error: {str(e)}"}})
            continue
        
        messages = response.get('Messages', [])

        if not messages:
            print("No messages present. Exiting program!")
            stop = True
            occupied = False
            create_instance_tag(instance_id, "status", "inactive")
            continue

        count=0
        
        Body = messages[0]['Body']
        ReceiptHandle = messages[0]['ReceiptHandle']

        body=json.loads(Body)

        user_id=body.get("user_id", "")
        if not dev:
            id=body.get("job_id", "")
        else:
            id=body.get("id", "")
        object_key=body.get("object_key", "")

        sqs.client.delete_message(
            QueueUrl=queue_prod if not dev else queue_dev,
            ReceiptHandle=ReceiptHandle
        )

        if not occupied:
            create_instance_tag(instance_id, "status", "occupied")

        stop_event=threading.Event()

        ping_thread=threading.Thread(
            target=ping,
            args=(instance_id, id, stop_event),
            daemon=True
        )

        ping_thread.start()

        if not dev:
            status, msg=s3.download_document_from_S3(id=id, user_id=user_id, object_key=object_key)
            if status:
                local_path=msg
            else:
                logging.info(msg)
                # Log the data in cosmos DB, if there is an error and continue
                cosmos.populate_container(items={"id":id, "user_id":user_id, "task":{"error":True, "msg":f"S3 error: {msg}"}})
                continue

        else:
            status, msg=azure.download_document_from_Azure(id=id, user_id=user_id)
            if status:
                local_path=msg
            else:
                logging.info(msg)
                cosmos.populate_container(items={"id":id, "user_id":user_id, "task":{"error":True, "msg":f"Azure error: {msg}"}})
                continue

        converted=False
        for root, dirs, files in os.walk(f"{parent_path}/raw_data/{id}"):
            for file in files:
                file=os.path.join(root, file)
                doc_path=file
                doc_name=doc_path.split("/")[-1]
                # if os.path.exists(f"{parent_path}/files/{doc_name.replace(doc_name.split('.')[-1], 'md')}"):
                #     continue
                # document.download_document_from_S3(id=job_id, user_id=client_id, doc_name=doc_name)
                if doc_name.lower().strip().endswith(("doc", "docx", "xls", "xlsx", "ppt", "pptx")):
                    try:
                        subprocess.run(
                            [
                                "libreoffice",
                                "--headless",
                                "--convert-to", "pdf",
                                "--outdir", root,
                                doc_path
                            ],
                            timeout=90
                        )
                    except subprocess.TimeoutExpired as timeout_expired:
                        cosmos.populate_container(items={"id":id, "user_id":user_id, "doc":doc_name, "task":{"error":True, "msg":"Conversion failed. Timeout expired!"}})
                        continue
                    except subprocess.CalledProcessError as process_error:
                        cosmos.populate_container(items={"id":id, "user_id":user_id, "doc":doc_name, "task":{"error":True, "msg":f"Conversion process error: {process_error}"}})
                        continue
                if not doc_name.lower().endswith("pdf"):
                    # logging.info(f'Original name: {doc_path}')
                    # new_source=f'{doc_path.replace(doc_name, doc_name.split(".")[:-1][0]+".pdf")}'
                    new_source=f'{doc_path.replace(doc_name, ".".join(doc_name.split(".")[:-1])+".pdf")}'
                    # logging.info(f'Converted to {new_source}')
                    converted=True
                    # markdown=document.convert_document_to_markdown(source=new_source, orig_source=doc_name, id=id, user_id=user_id, dev=dev)
                    # continue
                
                if converted:
                    markdown=document.convert_document_to_markdown(source=new_source, orig_source=doc_name, id=id, user_id=user_id, dev=dev)
                    # markdownSplitter.split(doc_id=id, source=new_source, markdown=markdown)
                    _startTime=time_in_ist()
                    cosmos.populate_container(items={
                        "id":id,
                        "user_id":user_id,
                        "doc":doc_name,
                        "task":{
                            "error":None,
                            "msg":None
                        },
                        "time":{
                            "start":_startTime,
                            "elapsed":"00:00:00",
                            "end":"00:00:00",
                            "tz":"Asia/Kolkata"
                        }
                    }, embedding=True)
                    # status, msg = start_embed(user_id=user_id, id=id, source=new_source, markdown=markdown)
                    status, msg = orchestrator.ingest_file(user_id=user_id, doc_id=id, source=os.path.basename(new_source), markdown=markdown)
                    _endTime=time_in_ist()
                    if status:
                        # Log the data to cosmosDB (embeddingLogs) and call the API for updating the frontend, if not dev
                        cosmos.delete_entry_and_update(items={
                            "id":id,
                            "user_id":user_id,
                            "doc":doc_name,
                            "task":{
                                "error":False,
                                "msg":"success"
                            },
                            "time":{
                                "start":_startTime,
                                "elapsed":elapsed_time(_startTime, _endTime),
                                "end":_endTime,
                                "tz":"Asia/Kolkata"
                            }
                        }, embedding=True)
                        if not dev:
                            # Call the API for updating the frontend (success)
                            # response=requests.post("https://api.digihumans.ai/app/doc-data-extraction/update-job-status",json={"user-id":user_id,"doc-id":id, "stage":"embedding", "jobSuccessful":True})
                            response=requests.post("https://api.digihumans.ai/app/knowledgebase/job-status-update",json={"type":"document", "id":id, "user-id":user_id, "stage":"embedding", "jobSuccessful":True})
                            logging.info(f"Response from save API: {response.text}")
                            sqs.send_success_to_sqs({"job_id":id})
                    else:
                        # Log the data to cosmosDB (embeddingLogs) and call the API for updating the frontend, if not dev
                        cosmos.delete_entry_and_update(items={
                            "id":id,
                            "user_id":user_id,
                            "doc":doc_name,
                            "task":{
                                "error":True,
                                "msg":msg
                            },
                            "time":{
                                "start":_startTime,
                                "elapsed":elapsed_time(_startTime, _endTime),
                                "end":_endTime,
                                "tz":"Asia/Kolkata"
                            }
                        }, embedding=True)
                        if not dev:
                            # Call the API for updating the frontend
                            # response=requests.post("https://api.digihumans.ai/app/doc-data-extraction/update-job-status",json={"user-id":user_id,"doc-id":id,"stage":"embedding","jobSuccessful":False})
                            response=requests.post("https://api.digihumans.ai/app/knowledgebase/job-status-update",json={"type":"document", "id":id, "user-id":user_id, "stage":"embedding", "jobSuccessful":False})
                            logging.info(f"Response from save API: {response.text}")
                            sqs.send_failure_to_sqs({"job_id":id})
                else:
                    markdown=document.convert_document_to_markdown(source=doc_path, id=id, user_id=user_id, dev=dev)
                    if markdown.startswith(("Processing error:", "Password protected Document")):
                        continue
                    _startTime=time_in_ist()
                    cosmos.populate_container(items={
                        "id":id,
                        "user_id":user_id,
                        "doc":doc_name,
                        "task":{
                            "error":None,
                            "msg":None
                        },
                        "time":{
                            "start":_startTime,
                            "elapsed":"00:00:00",
                            "end":"00:00:00",
                            "tz":"Asia/Kolkata"
                        }
                    }, embedding=True)
                    # status, msg = start_embed(user_id=user_id, id=id, source=doc_path, markdown=markdown)
                    status, msg = orchestrator.ingest_file(user_id=user_id, doc_id=id, source=os.path.basename(doc_path), markdown=markdown)
                    _endTime=time_in_ist()
                    if status:
                        # Log the data to cosmosDB (embeddingLogs) and call the API for updating the frontend, if not dev
                        cosmos.delete_entry_and_update(items={
                            "id":id,
                            "user_id":user_id,
                            "doc":doc_name,
                            "task":{
                                "error":False,
                                "msg":"success"
                            },
                            "time":{
                                "start":_startTime,
                                "elapsed":elapsed_time(_startTime, _endTime),
                                "end":_endTime,
                                "tz":"Asia/Kolkata"
                            }
                        }, embedding=True)

                        if not dev:
                            # Call the API for updating the frontend for embedding
                            # response=requests.post("https://api.digihumans.ai/app/doc-data-extraction/update-job-status",json={"user-id":user_id,"doc-id":id,"stage":"embedding","jobSuccessful":True})
                            response=requests.post("https://api.digihumans.ai/app/knowledgebase/job-status-update",json={"type":"document", "id":id, "user-id":user_id, "stage":"embedding", "jobSuccessful":True})

                    else:
                        # Log the data to cosmosDB (embeddingLogs) and call the API for updating the frontend, if not dev
                        cosmos.delete_entry_and_update(items={
                            "id":id,
                            "user_id":user_id,
                            "doc":doc_name,
                            "task":{
                                "error":True,
                                "msg":msg
                            },
                            "time":{
                                "start":_startTime,
                                "elapsed":elapsed_time(_startTime, _endTime),
                                "end":_endTime,
                                "tz":"Asia/Kolkata"
                            }
                        }, embedding=True)

                        if not dev:
                            # Call the API for updating the frontend for embedding
                            response=requests.post("https://api.digihumans.ai/app/knowledgebase/job-status-update",json={"type":"document", "id":id, "user-id":user_id, "stage":"embedding", "jobSuccessful":False})
                    # markdownSplitter.split(doc_id=id, source=doc_path, markdown=markdown)
        print("Stopping ping thread")
        stop_event.set()
        ping_thread.join()
