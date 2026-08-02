import logging

from fastapi import FastAPI

from trial_conversion_model.api.routes import router

# Configure logging once, here, so every prediction shows up in the logs.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(title="Beam trial conversion model")
app.include_router(router)
