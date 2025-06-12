from api import VideoGenerationAPI
from utilmeta_config import configure
import fastapi
from utilmeta import UtilMeta
from utilmeta.core import api


service = UtilMeta(
    __name__,
    name="heygem",
    description="API for generating videos from audio and video inputs.",
    host="0.0.0.0",
    backend=fastapi,
    asynchronous=True,
    port=8102,
)

configure(service)


@service.mount
@api.CORS(allow_origin="*")
class RootApi(api.API):
    video: VideoGenerationAPI


app = service.application()
if __name__ == "__main__":
    service.run()

