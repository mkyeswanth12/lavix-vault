from fastapi import FastAPI

from app.routers.chat.session_routes import router


def test_chat_crud_and_history_http_contract_remains_available():
    app = FastAPI()
    app.include_router(router, prefix="/api/ai")
    paths = app.openapi()["paths"]

    expected_methods = {
        "/api/ai/chats": {"get", "post"},
        "/api/ai/chats/{chat_id}": {"get", "patch", "delete"},
        "/api/ai/chat/sessions": {"get"},
        "/api/ai/chat/history": {"get", "delete"},
        "/api/ai/chat/history/{session_id}": {"delete"},
    }
    for path, methods in expected_methods.items():
        assert path in paths
        assert methods.issubset(paths[path])

    history_parameters = paths["/api/ai/chat/history"]["get"]["parameters"]
    session_id = next(parameter for parameter in history_parameters if parameter["name"] == "session_id")
    assert session_id["in"] == "query"
    assert session_id["required"] is False

    patch_schema = paths["/api/ai/chats/{chat_id}"]["patch"]["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert patch_schema["$ref"].endswith("/ChatPatchRequest")
