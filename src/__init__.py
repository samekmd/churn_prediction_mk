import logging
from dotenv import load_dotenv
import dagshub

load_dotenv()

# Initialize DagsHub with credentials
dagshub.init(
    repo_owner="samekmd",
    repo_name="churn_prediction_mk"
)

# Configure the logging strategy
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)
