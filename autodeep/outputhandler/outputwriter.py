import csv
import os


class OutputWriter:

    def __init__(self, filename, fields):
        """__init__

        Args:
        self : type
            Description
        filename : type
            Description
        fields : type
            Description

        Returns:
            type: Description
        """
        self.filename = filename
        self.fields = fields
        self._initialize_file()

    def _initialize_file(self):
        """_initialize_file

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
            with open(self.filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def write_row(self, **kwargs):
        """write_row

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        with open(self.filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            row = {}
            for field in self.fields:
                value = kwargs.get(field)
                if isinstance(value, (list, dict)):
                    value = str(value)
                row[field] = value
            writer.writerow(row)
