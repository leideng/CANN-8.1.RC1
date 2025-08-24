# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.


class ExportError(Exception):
    def __init__(self, message):
        super().__init__(message)  
        self.message = message

    def __str__(self):
        return f"ExportError: {self.message}"


class ParseError(Exception):
    def __init__(self, message):
        super().__init__(message) 
        self.message = message

    def __str__(self):
        return f"ParseError: {self.message}"


class MessageError(ParseError):
    pass


class DatabaseError(Exception):
    pass


class ValidationError(ParseError):
    def __init__(self, key, message="Failed to parse data"):
        super().__init__(message)
        self.key = key  
        self.message = message 

    def __str__(self):
        return f"{self.message}: {self.key}."


class KeyMissingError(ParseError):
    def __init__(self, key, message="Failed to parse data"):
        super().__init__(message)
        self.key = key

    def __str__(self):
        return f"{self.message}: {self.key} not exists."


class DataFrameMissingError(KeyMissingError):
    def __init__(self, key, message="Failed to read dataframe"):
        super().__init__(key, message)


class ColumnMissingError(KeyMissingError):
    def __init__(self, key, message="Failed to read column"):
        super().__init__(key, message)
        
    def __str__(self):
        return f"{self.message}: {self.key} not exists."


class LoadDataError(ParseError):
    def __init__(self, path, message="Failed to load data"):
        super().__init__(message)
        self.path = path  
        self.message = message  

    def __str__(self):
        # 返回详细的错误信息
        return f"{self.message}: {self.path}"


