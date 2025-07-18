import sys

from loguru import logger

from TinyTrain.global_var import LOGGING_NAME, RANK


# 统一过滤函数
def rank_filter(record):
    return record["extra"].get("rank", -1) in {-1, 0}

def set_logger(name: str = LOGGING_NAME):
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "format": "<cyan>{extra[log_name]}-[RANK {extra[rank]}]</> | <green>- {message}</>",
                "level": "INFO",
                "colorize": True,
                "filter": lambda r: r["level"].name == "INFO" and rank_filter(r),
            },
            {
                "sink": sys.stdout,
                "format": "<cyan>{extra[log_name]}-[RANK {extra[rank]}]</> | <yellow>- Warning: {message}</>",
                "level": "WARNING",
                "colorize": True,
                "filter": lambda r: r["level"].name == "WARNING" and rank_filter(r),
            },
            {
                "sink": sys.stdout,
                "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} |<lvl>{level:8}</>| {name} : {module}.py:{line:4} | <cyan>{extra[log_name]}-[RANK {extra[rank]}]</> | <lvl>- {message}</>",
                "level": "ERROR",
                "colorize": True,
                "filter": lambda r: r["level"].name == "ERROR" and rank_filter(r),
            },
            {
                "sink": sys.stdout,
                "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} |<lvl>{level:8}</>| {name} : {module}.py:{line:4} | <cyan>{extra[log_name]}-[RANK {extra[rank]}]</> | <lvl>- {message}</>",
                "level": "CRITICAL",
                "colorize": True,
                "filter": lambda r: r["level"].name == "CRITICAL" and rank_filter(r),
            },
        ]
    )

    new_logger = logger.bind(rank=RANK, log_name=name)

    return new_logger

