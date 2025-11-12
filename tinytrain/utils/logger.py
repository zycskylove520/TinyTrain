"""
Copyright (c) 2025 zycskylove520

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys

from loguru import logger

from tinytrain.global_var import LOGGING_NAME, RANK


# 统一过滤函数
def rank_filter(record):
    return record["extra"].get("rank", -1) in {-1, 0}

def set_logger(name: str = LOGGING_NAME):
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "format": "<cyan>{extra[log_name]}-[RANK: {extra[rank]}]</> | <green>- {message}</>",
                "level": "INFO",
                "colorize": True,
                "filter": lambda r: r["level"].name == "INFO" and rank_filter(r),
            },
            {
                "sink": sys.stdout,
                "format": "<cyan>{extra[log_name]}-[RANK: {extra[rank]}]</> | <yellow>- Warning: {message}</>",
                "level": "WARNING",
                "colorize": True,
                "filter": lambda r: r["level"].name == "WARNING" and rank_filter(r),
            },
            {
                "sink": sys.stdout,
                "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} |<lvl>{level:8}</>| {name} : {module}.py:{line:4} | <cyan>{extra[log_name]}-[RANK: {extra[rank]}]</> | <lvl>- {message}</>",
                "level": "ERROR",
                "colorize": True,
                "filter": lambda r: r["level"].name == "ERROR" and rank_filter(r),
            },
            {
                "sink": sys.stdout,
                "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} |<lvl>{level:8}</>| {name} : {module}.py:{line:4} | <cyan>{extra[log_name]}-[RANK: {extra[rank]}]</> | <lvl>- {message}</>",
                "level": "CRITICAL",
                "colorize": True,
                "filter": lambda r: r["level"].name == "CRITICAL" and rank_filter(r),
            },
        ]
    )

    new_logger = logger.bind(rank=RANK, log_name=name)

    return new_logger

