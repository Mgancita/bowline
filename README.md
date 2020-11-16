# Bowline
Configurable tools to easily pre and post process your data for data-science and machine learning.

## Quickstart
This will show you how to install and create a minimal implementation of `Bowline`. More exhaustive documentation is a main priority and will be released soon.

### Installation
```
$ pip install bowline
```

### Minimal implementation
```python
from bowline import StandardPreprocessor
import pandas as pd

raw_data = pd.read_csv('path/to/your/file')
preprocessor = StandardPrepreocessor(
    data = data,
    numerical_features = ["age", "capital-gain"],
    binary_features = ["sex"],
    categoric_features = ["occupation", "education"]
)
processed_data = preprocessor.process(target="sex")
```
