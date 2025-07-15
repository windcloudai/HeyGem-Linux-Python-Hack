import os
import shutil
import queue
import subprocess
import sys
import time
import traceback
import uuid
import jwt
from jwt import JWT, exceptions as jwt_exceptions
from jwt.jwk import OctetJWK

import cv2
from utilmeta.core import api, request, response
from utilmeta.core.file import File
from xengine_commons.api import APIStatus as AS, ResponseSchema as RS
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- 开始：添加 HeyGem-Linux-Python-Hack 到 sys.path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
hey_gem_hack_path = os.path.join(current_script_dir, "HeyGem-Linux-Python-Hack")
logging.info(f"current_script_dir: {current_script_dir}") # <-- 新增日志
logging.info(f"hey_gem_hack_path: {hey_gem_hack_path}") # <-- 新增日志

JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'fengyun-heygem-secret')
JWT_ALGORITHM = 'HS256'
JWK_KEY = OctetJWK(key=JWT_SECRET_KEY.encode())

def verify_jwt_token(token: str) -> bool:
    try:
        payload = JWT().decode(token, key=JWK_KEY, algorithms=[JWT_ALGORITHM])
        uid = payload['sub']
        if uid != "123456789":
            return False
        return True
    except jwt_exceptions.JWSDecodeError:
        return False

def jwt_required(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        auth_header = self.request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return RS.error(AS.PERMISSION_DENIED, "缺少认证令牌")
        
        token = auth_header.split(' ')[1]
        if not verify_jwt_token(token):
            return RS.error(AS.PERMISSION_DENIED, "无效的认证令牌")
            
        return await func(self, *args, **kwargs)
    return wrapper


if hey_gem_hack_path not in sys.path:
    sys.path.insert(0, hey_gem_hack_path)
# --- 结束：添加 HeyGem-Linux-Python-Hack 到 sys.path ---

# --- 开始：从 HeyGem-Linux-Python-Hack 导入实际模块 ---
# 假设您已经在 HeyGem-Linux-Python-Hack/service/__init__.py 和 y_utils/__init__.py 创建了文件
try:
    import service.trans_dh_service as actual_trans_dh_service
    from y_utils.config import GlobalConfig
    from y_utils.logger import logger as actual_logger # 避免与可能的其他logger冲突
    from h_utils.custom import CustomError as ActualCustomError # 假设h_utils也在HeyGem-Linux-Python-Hack下
    # 初始化配置，这非常重要，路径需要正确
    # GlobalConfig.instance().init_config(os.path.join(hey_gem_hack_path, "config/config.ini"))
    # 注意：您需要确保 GlobalConfig 被正确初始化，这通常在原始脚本的早期阶段完成。
    # 您可能需要找到 GlobalConfig 初始化的地方，并确保在API中也执行类似操作。
except ImportError as e:
    print(f"Error importing from HeyGem-Linux-Python-Hack: {e}")
    print(f"Please ensure HeyGem-Linux-Python-Hack is in sys.path: {sys.path}")
    print(f"And that necessary __init__.py files exist in subdirectories like 'service' and 'y_utils'.")
    # 在无法导入核心依赖时，可以选择退出或让后续代码因 NameError 失败
    # sys.exit("Failed to import core dependencies.") 
    # 或者，如果您有mock作为后备：
    actual_trans_dh_service = None # 或者指向您的 MockTransDhTask
    GlobalConfig = None          # 或者指向您的 MockGlobalConfig
    actual_logger = None         # 或者指向您的 MockLogger
    ActualCustomError = Exception # 基本的后备
    print("WARNING: Failed to import actual modules, API might not function correctly or use mocks.")

# --- 结束：从 HeyGem-Linux-Python-Hack 导入实际模块 ---

# 模拟 GlobalConfig，实际应从 y_utils.config 导入并初始化
class MockGlobalConfig:
    def __init__(self):
        self.watermark_path = "path/to/your/watermark.png"  # 示例路径
        self.digital_auth_path = "path/to/your/digital_auth.png" # 示例路径
        self.temp_dir = "temp_output" # 确保这个目录存在且可写
        self.result_dir = "final_output" # 确保这个目录存在且可写
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    @staticmethod
    def instance():
        if not hasattr(MockGlobalConfig, "_instance"):
            MockGlobalConfig._instance = MockGlobalConfig()
        return MockGlobalConfig._instance

# 模拟 logger，实际应从 y_utils.logger 导入
class MockLogger:
    def info(self, msg):
        print(f"INFO: {msg}")
    def error(self, msg):
        print(f"ERROR: {msg}")
logger = MockLogger()

# 模拟 CustomError
class CustomError(Exception):
    pass

# 从 run.py 移植并修改的 write_video 函数
# 主要修改：不再使用 queue，而是直接返回 result_path
# 移除了 exit(0)
def write_video_adapted(
    output_imgs_queue, # 在API场景下，这个队列的来源和填充方式需要重新设计
                       # 如果 work() 内部直接生成帧数据，则可能不需要外部队列
    temp_dir,
    result_dir,
    work_id,
    audio_path,
    # result_queue, # 移除队列
    width,
    height,
    fps,
    watermark_switch=0,
    digital_auth=0,
):
    output_mp4 = os.path.join(temp_dir, "{}-t.mp4".format(work_id))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result_path = os.path.join(result_dir, "{}-r.mp4".format(work_id))
    # 确保目录存在
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    logger.info("Custom VideoWriter init done")
    try:
        # ------------------------------------------------------------------
        # 关键修改点：output_imgs_queue 的处理
        # 在 run.py 中，这个队列由 TransDhTask().work() 的某个部分填充。
        # 在 API 环境中，您需要确保帧数据能被正确送入这里。
        # 为简化演示，我们假设队列会像 run.py 中那样被填充和结束。
        # 实际集成时，这部分逻辑最复杂，可能需要重构 TransDhTask。
        # ------------------------------------------------------------------
        while True: 
            # 模拟从队列获取数据，实际应由 TransDhTask().work() 内部逻辑驱动
            # state, reason, value_ = output_imgs_queue.get() 
            # ---- 模拟开始 ----
            # 此处为模拟，实际应由 work() 内部的图像处理逻辑填充
            # 假设处理完成，发送结束信号
            if output_imgs_queue.empty(): # 仅为模拟，实际结束条件不同
                 print("Simulating end of image queue")
                 state, reason, value_ = True, None, None 
            else:
                 state, reason, value_ = output_imgs_queue.get(timeout=1) # 加timeout防止永久阻塞
            # ---- 模拟结束 ----

            if isinstance(state, bool) and state is True:
                logger.info(
                    "Custom VideoWriter [{}]视频帧队列处理已结束".format(work_id)
                )
                logger.info(
                    "Custom VideoWriter Silence Video saved in {}".format(
                        os.path.realpath(output_mp4)
                    )
                )
                video_write.release()
                break
            else:
                if isinstance(state, bool) and not state:
                    logger.error(
                        "Custom VideoWriter [{}]任务视频帧队列 -> 异常原因:[{}]".format(
                            work_id, reason
                        )
                    )
                    raise CustomError(reason)
                for result_img in value_:
                    video_write.write(result_img)
        if video_write is not None:
            video_write.release()
        
        # FFMPEG 命令部分与 run.py 保持一致
        config = MockGlobalConfig.instance() # 使用模拟的Config
        if watermark_switch == 1 and digital_auth == 1:
            command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                audio_path, output_mp4, config.watermark_path, config.digital_auth_path, result_path)
        elif watermark_switch == 1 and digital_auth == 0:
            command = 'ffmpeg -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10" -c:a aac -crf 15 -strict -2 {}'.format(
                audio_path, output_mp4, config.watermark_path, result_path)
        elif watermark_switch == 0 and digital_auth == 1:
            command = 'ffmpeg -loglevel warning -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                audio_path, output_mp4, config.digital_auth_path, result_path)
        else:
            command = "ffmpeg -loglevel warning -y -i {} -i {} -c:a aac -c:v libx264 -crf 15 -strict -2 {}".format(
                audio_path, output_mp4, result_path)
        
        logger.info("Custom command: {}".format(command))
        subprocess.call(command, shell=True)
        logger.info("###### Custom Video Writer write over")
        logger.info(f"###### Video result saved in {os.path.realpath(result_path)}")
        return result_path

    except queue.Empty: # 处理队列为空的超时
        logger.error(f"Custom VideoWriter [{work_id}] timed out waiting for image queue.")
        if video_write is not None:
            video_write.release()
        return None # 或者抛出异常
    except Exception as e:
        logger.error(
            "Custom VideoWriter [{}]视频帧队列处理异常结束，异常原因:[{}]".format(
                work_id, e.__str__()
            )
        )
        if video_write is not None:
            video_write.release()
        raise # 重新抛出异常，让上层处理
    finally:
        logger.info("Custom VideoWriter 后处理进程结束")
        # 清理临时的无声视频
        if os.path.exists(output_mp4):
            try:
                os.remove(output_mp4)
                logger.info(f"Removed temporary file: {output_mp4}")
            except OSError as e:
                logger.error(f"Error removing temporary file {output_mp4}: {e}")

# 模拟 TransDhTask，实际应从 service.trans_dh_service 导入
# 这个模拟类需要能够调用上面适配过的 write_video_adapted
class MockTransDhTask:
    def __init__(self):
        # 模拟 run.py 中 service.trans_dh_service.write_video = write_video 的行为
        # 在实际的 TransDhTask 类中，它可能会在其内部方法中调用 self.write_video
        self.write_video_function = write_video_adapted 
        self.output_imgs_queue = queue.Queue() # 模拟的图像队列

    def work(self, audio_path, video_path, code, watermark_switch, digital_auth, param3, param4):
        logger.info(f"MockTransDhTask.work called with audio: {audio_path}, video: {video_path}")
        
        # 模拟视频处理过程，获取视频基本信息
        # 实际应用中，这些信息由 TransDhTask 内部的视频处理逻辑得到
        try:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if width == 0 or height == 0 or fps == 0:
                raise ValueError("Failed to get video properties from input video.")
        except Exception as e:
            logger.error(f"Error reading video properties: {e}")
            raise CustomError(f"Cannot process video: {video_path}. Details: {e}")

        # 模拟将图像帧放入队列的过程 (这是最需要适配的部分)
        # 在实际的 TransDhTask().work() 中，会有复杂的图像处理和帧生成逻辑
        # 这里我们只简单模拟一下，假设有一些处理后的帧
        # ---- 模拟帧处理开始 ----
        # 假设我们生成了一些空的图像帧放入队列
        # 实际上这里应该是调用人脸检测、3DMM转换等复杂逻辑
        num_mock_frames = min(int(fps * 5), frame_count) # 模拟5秒的视频帧
        mock_frame = cv2.imread("/home/ai/projects/fengyun-heygem/HeyGem-Linux-Python-Hack/1.jpeg") # 使用项目中的一个图片作为模拟帧
        if mock_frame is None:
            logger.error("Failed to load mock frame image. Ensure 1.jpeg exists.")
            # 创建一个黑色图像作为备用
            mock_frame = cv2.UMat(height, width, cv2.CV_8UC3) # 使用 UMat
            mock_frame[:] = (0,0,0) # 设置为黑色
        else:
            mock_frame = cv2.resize(mock_frame, (width, height))
            mock_frame = cv2.UMat(mock_frame) # 转换为 UMat
        
        for _ in range(num_mock_frames):
            self.output_imgs_queue.put((None, None, [mock_frame.get()])) # 放入实际的 numpy array
        self.output_imgs_queue.put((True, None, None)) # 发送结束信号
        # ---- 模拟帧处理结束 ----

        config = MockGlobalConfig.instance()
        # 调用适配后的 write_video 函数
        # 注意：这里的参数传递需要与 write_video_adapted 匹配
        result_video_path = self.write_video_function(
            self.output_imgs_queue, 
            config.temp_dir, 
            config.result_dir, 
            code, 
            audio_path, 
            width, 
            height, 
            fps, 
            watermark_switch, 
            digital_auth
        )
        
        if result_video_path and os.path.exists(result_video_path):
            return result_video_path
        else:
            logger.error("Video generation in MockTransDhTask failed.")
            return None

class VideoGenerationAPI(api.API):
    @api.post("/generate_video")
    @jwt_required
    async def generate_video(
        self,
        audio_file: File = request.BodyParam,
        video_file: File = request.BodyParam,
    ):
        if not actual_trans_dh_service or not GlobalConfig or not actual_logger:
            return response.Response("Core video processing modules are not available.", status=500)

        temp_upload_dir = "temp_uploads"
        os.makedirs(temp_upload_dir, exist_ok=True)

        # 使用唯一ID确保文件名不冲突
        request_id = uuid.uuid4().hex[:6]

        request_dir = os.path.join(current_script_dir, temp_upload_dir, request_id)

        await audio_file.asave(request_dir)
        await video_file.asave(request_dir)
        
        audio_path = os.path.join(request_dir, audio_file.filename)
        video_path = os.path.join(request_dir, video_file.filename)

        actual_logger.info(f"Uploaded audio to: {audio_path}")
        actual_logger.info(f"Uploaded video to: {video_path}")
        
        _ = GlobalConfig.instance() # Ensure instance is created/accessed
        actual_logger.info(f"Accessed GlobalConfig instance.")

        # --- Begin: Change CWD for submodule --- 
        # original_cwd = os.getcwd()
        # logging.info(f"Original CWD: {original_cwd}")
        # os.chdir(hey_gem_hack_path)
        # logging.info(f"Attempted to change CWD to: {hey_gem_hack_path}")
        # logging.info(f"CWD after chdir: {os.getcwd()}")
        
        # +++ 新增调试代码 +++
        model_path_to_check = 'wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt'
        absolute_model_path = os.path.abspath(model_path_to_check)
        path_exists = os.path.exists(model_path_to_check)
        logging.info(f"Checking for model at relative path: '{model_path_to_check}'")
        logging.info(f"Absolute path resolved to: '{absolute_model_path}'")
        logging.info(f"Does path exist (from api.py perspective after chdir)? {path_exists}")
        # +++ 结束新增调试代码 +++

        # --- End: Change CWD for submodule --- 

        try:
            # Instantiate真实的 TransDhTask
            if hasattr(actual_trans_dh_service, "TransDhTask"):
                logging.info("TransDhTask init (attempting instantiation)") # 修改日志以区分
                task_instance = actual_trans_dh_service.TransDhTask()
            else:
                actual_logger.error("TransDhTask class not found in imported service.trans_dh_service")
                return RS.error(AS.INTERNAL_SERVER_ERROR, "Video processing task handler not found.")

            # 原始 run.py 中有 service.trans_dh_service.write_video = write_video = write_video
            # 这意味着 TransDhTask 内部可能依赖一个全局的 write_video 函数。
            # 您需要确保这个依赖在 API 环境中也被满足。最干净的方式是修改 TransDhTask
            # 使其不依赖全局赋值，而是将 write_video 作为方法或参数传入。
            # 如果不能修改，您可能需要在 API 初始化时也执行类似的赋值，
            # 但这会使代码耦合度增高。
            # 假设 TransDhTask 内部已经处理了视频写入逻辑，或者 write_video 已被正确设置。

            task_instance.work(
                audio_path, # This path is absolute or relative to original_cwd
                video_path, # This path is absolute or relative to original_cwd
                request_id,
                0, # watermark_switch
                0, # digital_auth
                0, # param3 (example, adjust as needed)
                0  # param4 (example, adjust as needed)
            )
            
            output_video_path = f"{request_id}-r.mp4"

            if output_video_path and os.path.exists(output_video_path):
                result_path = f"{request_id}.mp4"
                shutil.move(output_video_path, os.path.join("output", result_path))
                
                # remove temp files
                os.remove(f"{request_id}_format.mp4")
                os.remove(f"{request_id}_format.wav")
                os.remove(f"{request_id}-t.mp4")
                
                # remove input data
                shutil.rmtree(request_dir)

                actual_logger.info(f"Video generated successfully: {result_path}")
                return RS.success(data={"video": result_path})
            else:
                actual_logger.error("Video generation failed or output file not found after task.work.")
                return RS.error(AS.INTERNAL_SERVER_ERROR, "Video generation failed or output file not found.")

        except ActualCustomError as e:
            actual_logger.error(f"CustomError during video generation: {e}")
            return RS.error(AS.INTERNAL_SERVER_ERROR, f"Video generation error: {str(e)}")
        except Exception as e:
            actual_logger.error(f"Unexpected error during video generation: {e} - {traceback.format_exc()}")
            return RS.error(AS.INTERNAL_SERVER_ERROR, f"Video generation failed due to an unexpected error: {str(e)}")
        finally:
            # --- Begin: Revert CWD --- 
            # os.chdir(original_cwd)
            # actual_logger.info(f"Reverted CWD to: {original_cwd}")
            # --- End: Revert CWD --- 
            
            # Ensure paths are defined before trying to remove them
            # These paths were created relative to the original CWD or are absolute.
            audio_path_to_remove = locals().get('audio_path')
            video_path_to_remove = locals().get('video_path')
            if audio_path_to_remove and os.path.exists(audio_path_to_remove): 
                os.remove(audio_path_to_remove)
            if video_path_to_remove and os.path.exists(video_path_to_remove): 
                os.remove(video_path_to_remove)

