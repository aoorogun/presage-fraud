import uvicorn
from presage.serving.app import create_app

if __name__ == "__main__":
    import os
    uvicorn.run(create_app(), host=os.environ.get("PRESAGE_HOST", "127.0.0.1"), port=int(os.environ.get("PRESAGE_PORT", "8000")))
