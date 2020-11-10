import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split

import config

r = requests.get(config.URL)

soup = BeautifulSoup(r.content, features="lxml")


def clean_formula(formula):
    formula = str(formula).replace("<td>", "").replace("<sub>", "")
    formula = formula.replace("</td>", "").replace("</sub>", "")
    clean_f = []
    for char in formula:
        if char == "&" or char == " ":
            break
        clean_f.append(char)
    return "".join(clean_f).strip()


data = {
    "formula": [],
    "name": [],
}

for table in soup.find_all("table", {"class": "wikitable"}):
    for row in table.find_all("tr"):
        val = row.find_all("td")
        if val and len(val) == 3:
            formula = clean_formula(val[0])
            formula = formula if "<" not in formula else None

            name = val[1].string
            if name is None:
                name = val[1].find("a")["title"]

            if formula and name:
                data["formula"].append(formula)
                data["name"].append(name.lower())

df = pd.DataFrame(data)

train, val = train_test_split(df, test_size=0.1)
train.to_csv(config.TRAIN_PATH, index=False, header=False)
val.to_csv(config.VALID_PATH, index=False, header=False)
