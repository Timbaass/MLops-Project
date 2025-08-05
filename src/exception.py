import sys
import logging
from logger import setup_logging

setup_logging()

def error_message_details(error, error_detail: sys):
    """Extracts error details from the exception."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )
    return error_message

class CustomException(Exception):
    """Custom exception class for handling exceptions in the application."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        """Returns the string representation of the error message."""
        logging.info(self.error_message)
        return self.error_message
