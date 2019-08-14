import csv
import datetime
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# Downloaded from https://www.bart.gov/about/reports/ridership
SOURCE_FILE = "bart-SFIA-EMBR-2011.csv"
DESTIN_FILE = "bart-SFIA-EMBR-2011.pkl"


if __name__ == "__main__":
    # Load csv file.
    start_date = datetime.datetime.strptime("2011-01-01", "%Y-%m-%d")
    dates = []
    counts = []
    with open(os.path.join(DATA, SOURCE_FILE)) as f:
        for date, hour, origin, destin, trip_count in csv.reader(f):
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
            date += datetime.timedelta(hours=int(hour))
            dates.append(int((date - start_date).total_seconds() / 3600))
            counts.append(int(trip_count))
    print("Loaded {} dense counts".format(len(dates)))
    assert 0 <= min(dates) < 24

    # Convert to PyTorch.
    result = torch.zeros(1 + max(dates))
    for date, count in zip(dates, counts):
        result[date] = count

    print("Saving {} dense counts".format(len(result)))
    torch.save(result, os.path.join(DATA, DESTIN_FILE))
