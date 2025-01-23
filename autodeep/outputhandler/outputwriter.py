import csv


class OutputWriter:
    def __init__(self, filename, fields):
        self.filename = filename
        self.fields = fields

    def write_row(self, **kwargs):  # Add **kwargs to capture additional arguments
        with open(self.filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if f.tell() == 0:
                writer.writeheader()
            row = {}
            for field in self.fields:
                value = kwargs.get(field)  # Access values from kwargs
                if isinstance(value, (list, dict)):
                    value = str(value)  # Convert lists or dicts to strings
                row[field] = value
            writer.writerow(row)
