# Formula to Name
Predicts the chemical name of any (real or not) chemical formula.

![sample_image](https://github.com/rishabhvarshney14/Formula-to-Name/blob/main/image/sample.PNG)

Check out the Live App [here](https://formula-to-name.herokuapp.com/).

This scrape data from [Wikipedia](https://en.wikipedia.org/wiki/Glossary_of_chemical_formulae) for training.

## How to Use
1. Create a virtual environment and activate it
 ```
 virtualenv formula-env
 source env/bin/activate
 ```
 
 2. Install required modules
 ```pip install -r requirements.txt ```
 
 3. Build the dataset (for custom dataset add train.csv and valid.csv in data folder)
 ```python src/get_data.py```
 
 4. Train the model
 ```python src/train.py```

You can give args to train.py
 
 | args                | shortcut | type  | example | default |
|---------------------|----------|-------|---------|---------|
| --learning_rate     | -lr      | float | 0.02    | 0.0003  |
| --epochs            | --       | int   | 10      | 10      |
| --continue_training | --       | bool  | True    | False   |

For example

```python src/train.py -lr 0.001 --continue_training True```
 
5. Test the model
 ```python src/test.py <formula>```
 
 for example 
 ```python src/test.py NaOH```
