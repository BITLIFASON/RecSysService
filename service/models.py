import typing as tp

from pydantic import BaseModel


class Error(BaseModel):
    error_key: tp.Union[tp.Any, str, None]
    error_message: tp.Union[tp.Any, str, None]
    error_loc: tp.Optional[tp.Any] = None
