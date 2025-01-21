import csv
import os
import joblib


class OutputWriter:
    def __init__(self, filename, fields):
        self.filename = filename
        self.fields = fields

    def write_row(self, **kwargs):
        with open(self.filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if f.tell() == 0:
                writer.writeheader()
            row = {}
            for field in self.fields:
                value = kwargs.get(field)
                if isinstance(value, (list, dict)):
                    value = str(value)
                row[field] = value
            writer.writerow(row)
