import logging


formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("maploc")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(handler)
logger.propagate = False
