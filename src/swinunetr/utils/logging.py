import logging
import sys
import os
from logging.handlers import RotatingFileHandler

class Logger:
    
    def __init__(
        self, 
        log_dir: str = "logs", 
        filename: str = "app.log", 
        console_level: int = logging.INFO, 
        file_level: int = logging.DEBUG
    ):
        """
        Initializes and configures the logger.
        
        Args:
            project_name (str): The name of the root logger for the project. 
                                This is mandatory for hierarchical logging.
            log_dir (str): Directory to save log files.
            filename (str): Name of the log file.
            console_level (int): The minimum level to output to the console.
            file_level (int): The minimum level to write to the file.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.project_name = 'project'
        self.log_path = os.path.join(log_dir, filename)
        self.console_level = console_level
        self.file_level = file_level

        # Get and configure the main project logger
        self.logger = logging.getLogger(self.project_name)
        
        # Set the logger's own level to the lowest of the two handlers.
        # This acts as a pre-filter before logs are passed to handlers.
        self.logger.setLevel(min(console_level, file_level))

        # 4. Prevent duplicate handlers if already configured
        if self.logger.hasHandlers():
            self.logger.info("Logger is already configured. Skipping setup.")
            return

        # 5. Create a common formatter
        self.formatter = self._create_formatter()
        
        # 6. Add handlers (delegated to private methods)
        self._add_console_handler()
        self._add_file_handler()

        self.logger.info(f"Logging setup complete. Log file: {self.log_path}")

    def _create_formatter(self) -> logging.Formatter:
        """
        (Helper) Creates and returns a standard log formatter.
        """
        log_format = (
            "%(asctime)s [%(levelname)-8s] %(name)s "
            "(%(filename)s:%(lineno)d): %(message)s"
        )
        return logging.Formatter(log_format)

    def _add_console_handler(self):
        """
        (Helper) Creates and adds the console (StreamHandler) handler.
        This handler prints logs to the terminal.
        """
        # Create console handler (StreamHandler) for stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level) # Set console-specific level
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """
        (Helper) Creates and adds the file (RotatingFileHandler) handler.
        This handler writes logs to a file and rotates it when it gets too large.
        """
        # Create file handler (RotatingFileHandler)
        # Rotates files when they reach 5MB, keeps 5 backup files.
        file_handler = RotatingFileHandler(
            self.log_path, 
            maxBytes=5*1024*1024, # 5 MB
            backupCount=5, 
            encoding='utf-8'
        )
        file_handler.setLevel(self.file_level) # Set file-specific level
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.
        Note: It's generally better to use logging.getLogger(project_name) directly.
        """
        return self.logger

