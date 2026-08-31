# single-file-data-processing


example:
```py
from orchestrator import Orchestrator
from configparser import ConfigParser
import os

config = ConfigParser()

config.read(".config")

api_key = config.get("api_settings", "api_key")

file = "dataset/winequality-white.csv"
orchestrator = Orchestrator(api_key, "gpt-5.6-luna")
orchestrator.preprocess(file)
 ```

```sh
preprocessor dataset/winequality-white.csv
```