# # app/core/config.py
# from pathlib import Path
# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     MODEL_PATH: str = str(Path("model.pt"))
#     IMG_SIZE: int = 640                    
#     CONF_THRESH: float = 0.3                
#     MAX_DET: int = 300                     
    
# def get_settings() -> Settings:
#     return Settings()

# app/core/config.py (unchanged)
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = str(Path("model.pt"))
    IMG_SIZE: int = 640
    CONF_THRESH: float = 0.3
    MAX_DET: int = 300

def get_settings() -> Settings:
    return Settings()
