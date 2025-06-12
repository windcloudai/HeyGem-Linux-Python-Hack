from utilmeta import UtilMeta
from utilmeta.core.server.backends.django import DjangoSettings
from utilmeta.core.orm import Database
from utilmeta.ops import Operations

INPUT_DIR = "input/"
OUTPUT_DIR = "output/"


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


