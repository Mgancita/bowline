# Bowline

![build](https://github.com/mgancita/bowline/workflows/Continuous%20Integration/badge.svg)
[![PyPI version](https://badge.fury.io/py/bowline.svg)](https://badge.fury.io/py/bowline)
![versions](https://img.shields.io/pypi/pyversions/bowline.svg)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
![PyPI downloads](https://img.shields.io/github/downloads/mgancita/bowline/total)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Configurable tools to easily pre and post process your data for data-science and machine learning.

## Quickstart
This will show you how to install and create a minimal implementation of `Bowline`. For more in-depth examples visit the [Official Docs](https://mgancita.github.io/bowline).

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
