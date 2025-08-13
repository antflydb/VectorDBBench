class AntflyError(Exception):
    """Base exception for Antfly client errors."""
    pass

class AntflyConnectionError(AntflyError):
    """Raised for connection-related errors."""
    pass

class AntflyHTTPError(AntflyError):
    """Raised for HTTP errors."""
    def __init__(self, message, status_code=None, response_text=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_text = response_text

class AntflyIndexError(AntflyError):
    """Raised for index-related errors."""
    pass

class AntflyTableError(AntflyError):
    """Raised for table-related errors."""
    pass

class AntflyValidationError(AntflyError):
    """Raised for validation errors."""
    pass
