import csv
import os


class OutputWriter:
    def __init__(self, filename, fields):
        self.filename = filename
        self.fields = fields
        self._initialize_file()

    def _initialize_file(self):
        """Ensure the file has a header if it doesn't exist."""
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
            with open(self.filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def write_row(self, **kwargs):
        with open(self.filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            row = {}
            for field in self.fields:
                value = kwargs.get(field)  # Access values from kwargs
                if isinstance(value, (list, dict)):
                    value = str(value)  # Convert lists or dicts to strings
                row[field] = value
            writer.writerow(row)
