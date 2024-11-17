from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


class RegularDataProcessor:
    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols if num_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.num_pipeline = Pipeline([("scaler", StandardScaler())])
        self.cat_pipeline = Pipeline(
            [("encoder", OneHotEncoder(handle_unknown="ignore"))]
        )
        self._build_processor()

    def _build_processor(self):
        transformers = []
        if self.num_cols:
            transformers.append(("num", self.num_pipeline, self.num_cols))
        if self.cat_cols:
            transformers.append(("cat", self.cat_pipeline, self.cat_cols))
        self.processor = ColumnTransformer(transformers)

    def fit(self, data):
        self.processor.fit(data)
        self._types = data.dtypes  # Store original data types
        self._col_order_ = data.columns  # Store original column order

        # Store scaling and encoding information
        if self.num_cols:
            self._num_mean = (
                self.processor.named_transformers_["num"].named_steps["scaler"].mean_
            )
            self._num_std = (
                self.processor.named_transformers_["num"].named_steps["scaler"].scale_
            )
            print(self._num_mean)
            print(self._num_std)
        if self.cat_cols:
            self._cat_categories = (
                self.processor.named_transformers_["cat"]
                .named_steps["encoder"]
                .categories_
            )
            print(self._cat_categories)

    def process_data(self, data):
        return self.processor.transform(data)
