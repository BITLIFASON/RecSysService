# type: ignore

import typing

from fastapi import APIRouter, FastAPI, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from service.api.exceptions import AuthorizationError, ModelNotFoundError, UserNotFoundError
from service.credentials import API_KEY
from service.log import app_logger
from service.models import Error
from service.recmodels import ann_als, popular, simple_range, user_knn


class RecoResponse(BaseModel):
    user_id: int
    items: typing.List[int]


error_sample = {"error_key": "string", "error_message": "string", "error_loc": "string"}
responses = {
    200: {"description": "Success", "model": Error},
    404: {"description": "Model or user is unknown", "model": Error},
    401: {"description": "API key is invalid", "model": Error},
    403: {"description": "Not authenticated", "model": Error},
}


router = APIRouter()
api_key_header = APIKeyHeader(name="Authorization")


def verify_token(token: str = Security(api_key_header)) -> str:
    if token == API_KEY:
        return token
    raise AuthorizationError()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@typing.no_type_check
@router.get(
    path="/reco/{model_name}/{user_id}", tags=["Recommendations"], response_model=RecoResponse, responses=responses
)
async def get_reco(
    request: Request, model_name: str, user_id: int, token: str = Security(verify_token)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name == "simple_range":
        reco = simple_range.predict()
    elif model_name == "popular":
        popular.load()
        reco = popular.predict()
    elif model_name == "user_knn":
        user_knn.load()
        reco = user_knn.predict(user_id)
    elif model_name == "ANN_ALS":
        ann_als.load()
        reco = ann_als.predict(user_id)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
