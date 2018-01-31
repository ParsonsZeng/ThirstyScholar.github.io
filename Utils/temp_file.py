import tempfile
import os

import pandas as pd
import numpy as np


temp_file = tempfile.NamedTemporaryFile()
print(temp_file)
print(temp_file.name)

# Create some data
dates = pd.date_range('20180101', periods=5)
df = pd.DataFrame(np.random.randn(5,4),
                  index=dates, columns=['a','b','c','d'])
print(df.head())

# Save to temp file
df.to_csv(temp_file.name)

# Read from temp file
df = pd.read_csv(temp_file.name, index_col=[0])
print(df.head())

temp_file.close()
