import os
from datetime import datetime
import shutil
import json
import logging


class CodeLogger:
    def __init__(self, comment=None, args=None, file_name=None):
        self.run_time = datetime.now()
        self.time_fmt = "%Y-%m-%d-%H-%M-%S"
        # self.time_fmt = "%Y%m%d%H%M%S"
        self.comment = comment
        self.log_dir = self.run_time.strftime(self.time_fmt)
        if comment is not None:
            self.log_dir = f"{self.log_dir}-{comment}"
        self.log_dir = os.path.join("../log", self.log_dir)
        self.args = args

        # self.args['_file_name'] = file_name

        # create log dir if not exists
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        # back up codes
        shutil.copytree("../code/", os.path.join(self.log_dir, "code/"))
        if self.args is not None:
            self.save_args(args)
        self.logger = self.init_logger()

    def save_args(self, args):
        with open(os.path.join(self.log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f)

    def init_logger(self):
        logger = logging.getLogger(self.comment)
        logger.setLevel(logging.INFO)
        log_format = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s")
        filename = os.path.join(self.log_dir, "run.log")
        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        logger.addHandler(log_handler)
        console_logger = logging.StreamHandler()
        console_logger.setLevel(logging.INFO)
        console_logger.setFormatter(log_format)
        logger.addHandler(console_logger)
        return logger
