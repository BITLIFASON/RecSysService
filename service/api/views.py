from typing import List

from fastapi import APIRouter, FastAPI, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from service.api.exceptions import UserNotFoundError, AuthorizationError, ModelNotFoundError
from service.log import app_logger
from service import recmodels
from service.credentials import API_KEY

class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
api_key_header = APIKeyHeader(name='X-SECRET')


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


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
        request: Request,
        model_name: str,
        user_id: int,
        token: str = Security(verify_token)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    if model_name == 'simple_range':
        reco = recmodels.simple_range(k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
