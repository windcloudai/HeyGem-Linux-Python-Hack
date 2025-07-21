from utilmeta import UtilMeta
from utilmeta.core.server.backends.django import DjangoSettings
from utilmeta.core.orm import Database
from utilmeta.ops import Operations

INPUT_DIR = "input/"
OUTPUT_DIR = "output/"

TTS_URL = "http://10.0.74.56:8102/speech/tts"
TTS_OUTPUT = "http://10.0.74.56:8120/speech_output"
TTS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiAiMTIzNDU2Nzg5IiwgIm5hbWUiOiAiZmVuZ3l1bi1tZWdhdHRzLXVzZXIifQ.JviECs8wUP1LmVgVu0AB5TBlWjEOisxnpwdWN_oReGc"


def configure(service: UtilMeta):
    service.use(Operations(
        route='ops',
        database=Database(
            name='heygem_ops'
        ),
        connection_key="123",
        base_url="https://ai.59.local/digital-human/seech",
        private_scope=["*"]
    ))
    service.use(DjangoSettings(
        secret_key='YOUR_SECRET_KEY',
        allowed_hosts=['localhost', "10.0.74.59", "ai.59.local"]
    ))
    service.setup()


