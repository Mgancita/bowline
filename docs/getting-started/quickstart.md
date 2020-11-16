# Quickstart

## Overview 
Bowline is split up into *Preprocessors* and *PostProcessors* and are stored in `bowline.preprocessors` and `bowline.postprocessors`, respectively. They're all made with a simple implementation as the default but contain extensive advanced configurations for processing. Lets start with the simplest one.

## Load in processor
Lets load in the `StandardPreprocessor`.
```python
from bowline.preprocessors import StandardPreprocessor
```

## Process Data
Now we read in a csv, pass it to `StandardPreprocessor` with some dataset specific information, and finally process the dataset for a given `target` variable.
```python
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

## Further configuration
All Preprocessors and Postprocessors have additional parameters to configure how the data is processed. Currently, there is no official documentation for it but each class has extensive docstings to help with implementation.
