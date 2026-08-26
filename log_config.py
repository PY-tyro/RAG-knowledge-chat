"""
日志配置模块

集中配置 logging，让各模块的日志统一格式、统一级别，替代原来零散的 print 调试输出。

用法：
    在程序入口（app_qa.py / app_file_uploader.py）中：
        from log_config import setup_logging
        setup_logging()

默认级别为 INFO（只输出 INFO/WARNING/ERROR）；需要排查检索或 Prompt 拼接问题时，
把 level 改成 logging.DEBUG 即可看到完整 Prompt 等调试信息。
"""

import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
