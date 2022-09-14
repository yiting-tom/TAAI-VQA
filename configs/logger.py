#%%
import logging
import time
from configs import pathes
from pathlib import Path

log_dir: Path = pathes.d_LOGS

# create log directory if not exists
log_dir.mkdir(parents=True, exist_ok=True)

# set log file name
log_file_name = f"{time.strftime('%Y%m%d_%H%M%S')}.log"

l = logging.getLogger('l')
l.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(
    filename=log_dir / log_file_name,
    mode='w'
)
streamHandler = logging.StreamHandler()

allFormatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
)

fileHandler.setFormatter(allFormatter)
fileHandler.setLevel(logging.INFO)

streamHandler.setFormatter(allFormatter)
streamHandler.setLevel(logging.INFO)

l.addHandler(streamHandler)
l.addHandler(fileHandler)