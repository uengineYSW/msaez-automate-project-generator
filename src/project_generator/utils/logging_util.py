import logging
import sys
from typing import Optional
import traceback

from ..config import Config

class LoggingUtil:
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """환경별 설정이 적용된 로거 반환"""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            
            # 핸들러가 이미 있으면 제거 (중복 방지)
            if logger.handlers:
                logger.handlers.clear()
            
            # 핸들러 설정
            handler = logging.StreamHandler(sys.stdout)
            
            # 환경별 포맷 설정
            if Config.is_local_run():
                # 로컬: 상세한 포맷
                formatter = logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            else:
                # Pod: 간결한 포맷 (Kubernetes 로그에 적합)
                formatter = logging.Formatter(
                    '[%(levelname)s] [%(name)s] %(message)s'
                )
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 로그 레벨 설정
            log_level = getattr(logging, Config.get_log_level().upper(), logging.INFO)
            logger.setLevel(log_level)
            
            # 상위 로거로 전파 방지
            logger.propagate = False
            
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def debug(cls, logger_name: str, message: str, pod_id: Optional[str] = None):
        """디버그 로그 (로컬에서만 출력)"""
        logger = cls.get_logger(logger_name)
        if pod_id:
            message = f"[{pod_id}] {message}"
        logger.debug(message)
    
    @classmethod 
    def info(cls, logger_name: str, message: str, pod_id: Optional[str] = None):
        """정보 로그"""
        logger = cls.get_logger(logger_name)
        if pod_id:
            message = f"[{pod_id}] {message}"
        logger.info(message)
    
    @classmethod
    def warning(cls, logger_name: str, message: str, pod_id: Optional[str] = None):
        """경고 로그"""
        logger = cls.get_logger(logger_name)
        if pod_id:
            message = f"[{pod_id}] {message}"
        logger.warning(message)
    
    @classmethod
    def error(cls, logger_name: str, message: str, pod_id: Optional[str] = None):
        """에러 로그"""
        logger = cls.get_logger(logger_name)
        if pod_id:
            message = f"[{pod_id}] {message}"
        logger.error(message)
    
    @classmethod
    def exception(cls, logger_name: str, message: str, exception: Exception, pod_id: Optional[str] = None):
        """예외 로그"""
        logger = cls.get_logger(logger_name)
        if pod_id:
            message = f"[{pod_id}] {message} {exception} {traceback.format_exc()}"
        else:
            message = f"{message}, {exception}, {traceback.format_exc()}"
        logger.error(message)