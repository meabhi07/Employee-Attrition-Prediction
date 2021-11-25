```python
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
```


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MMM-YY</th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
      <th>Dateofjoining</th>
      <th>LastWorkingDate</th>
      <th>Joining Designation</th>
      <th>Designation</th>
      <th>Total Business Value</th>
      <th>Quarterly Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19099</th>
      <td>2017-08-01</td>
      <td>2788</td>
      <td>30</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>2017-06-08</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>740280</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19100</th>
      <td>2017-09-01</td>
      <td>2788</td>
      <td>30</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>2017-06-08</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>448370</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19101</th>
      <td>2017-10-01</td>
      <td>2788</td>
      <td>30</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>2017-06-08</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19102</th>
      <td>2017-11-01</td>
      <td>2788</td>
      <td>30</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>2017-06-08</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>200420</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19103</th>
      <td>2017-12-01</td>
      <td>2788</td>
      <td>30</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>2017-06-08</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>411480</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
train['LastWorkingDate'] = train['LastWorkingDate'].fillna(0)
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MMM-YY</th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
      <th>Dateofjoining</th>
      <th>LastWorkingDate</th>
      <th>Joining Designation</th>
      <th>Designation</th>
      <th>Total Business Value</th>
      <th>Quarterly Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>2015-12-24</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2381060</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-02-01</td>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>2015-12-24</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-665480</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-01</td>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>2015-12-24</td>
      <td>2016-03-11</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-11-01</td>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
      <td>2017-11-06</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-12-01</td>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
      <td>2017-11-06</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame({'Emp_ID':sorted(train.Emp_ID.unique())})
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
age_col = train.groupby(['Emp_ID']).Age.agg(['min'])
age_col = age_col.reset_index()
age_col.columns = ['Emp_ID','Age']

def get_age(x):
  return age_col[age_col.Emp_ID==x].Age.iloc[0]

df["Age"] = age_col.Emp_ID.apply(get_age)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 2 columns</p>
</div>




```python
gen_col = train.groupby(['Emp_ID','Gender']).Gender.agg('count').to_frame()
gen_col.columns = ['gen_count']
gen_col = gen_col.reset_index()

def get_gen(x):
  return gen_col[gen_col.Emp_ID==x].Gender.iloc[0]

df["Gender"] = gen_col.Emp_ID.apply(get_gen)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 3 columns</p>
</div>




```python
city_col = train.groupby(['Emp_ID','City']).City.agg('count').to_frame()
city_col.columns = ['city_count']
city_col = city_col.reset_index()

def get_city(x):
  return city_col[city_col.Emp_ID==x].City.iloc[0]

df["City"] = city_col.Emp_ID.apply(get_city)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 4 columns</p>
</div>




```python
edu_col = train.groupby(['Emp_ID','Education_Level']).Education_Level.agg('count').to_frame()
edu_col.columns = ['edu_count']
edu_col = edu_col.reset_index()

def get_edu(x):
  return edu_col[edu_col.Emp_ID==x].Education_Level.iloc[0]

df["Education_Level"] = edu_col.Emp_ID.apply(get_edu)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
      <td>Master</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
      <td>College</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
      <td>Bachelor</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 5 columns</p>
</div>




```python
sal_col = train.groupby(['Emp_ID']).Salary.agg(['max'])
sal_col = sal_col.reset_index()
sal_col.columns = ['Emp_ID','Salary']

def get_sal(x):
  return sal_col[sal_col.Emp_ID==x].Salary.iloc[0]

df["Salary"] = sal_col.Emp_ID.apply(get_sal)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
      <td>Master</td>
      <td>65603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
      <td>College</td>
      <td>46368</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
      <td>Bachelor</td>
      <td>78728</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 6 columns</p>
</div>




```python
all_occur = train.groupby(['Emp_ID']).size().to_frame()
all_occur = all_occur.reset_index()
all_occur.columns = ['Emp_ID','months_worked']
def months_worked_baby(x):
  return all_occur[all_occur.Emp_ID==x].months_worked.iloc[0]

df["months_worked"] = all_occur.Emp_ID.apply(months_worked_baby)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
      <th>months_worked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
      <td>Master</td>
      <td>65603</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
      <td>College</td>
      <td>46368</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
      <td>Bachelor</td>
      <td>78728</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 7 columns</p>
</div>




```python
join_des = train.groupby(['Emp_ID'])['Joining Designation'].agg(['max'])
join_des = join_des.reset_index()
join_des.columns = ['Emp_ID','Joining_Designation']
join_des

def get_jdes(x):
  return join_des[join_des.Emp_ID==x].Joining_Designation.iloc[0]

df["Joining_Designation"] = join_des.Emp_ID.apply(get_jdes)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
      <th>months_worked</th>
      <th>Joining_Designation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
      <td>Master</td>
      <td>65603</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
      <td>College</td>
      <td>46368</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
      <td>Bachelor</td>
      <td>78728</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>7</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 8 columns</p>
</div>




```python
cur_des = train.groupby(['Emp_ID'])['Designation'].agg(['max'])
cur_des = cur_des.reset_index()
cur_des.columns = ['Emp_ID','Designation']
cur_des
def get_cdes(x):
  return cur_des[cur_des.Emp_ID==x].Designation.iloc[0]

df["Designation"] = cur_des.Emp_ID.apply(get_cdes)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
      <th>months_worked</th>
      <th>Joining_Designation</th>
      <th>Designation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
      <td>Master</td>
      <td>65603</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
      <td>College</td>
      <td>46368</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
      <td>Bachelor</td>
      <td>78728</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>24</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 9 columns</p>
</div>




```python
qua_rat = train.groupby(['Emp_ID'])['Quarterly Rating'].agg(['min'])
qua_rat = qua_rat.reset_index()
qua_rat.columns = ['Emp_ID','Minimum_Quarterly_Rating']
qua_rat

def get_rat(x):
  return qua_rat[qua_rat.Emp_ID==x].Minimum_Quarterly_Rating.iloc[0]

df["Minimum_Quarterly_Rating"] = qua_rat.Emp_ID.apply(get_rat)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
      <th>months_worked</th>
      <th>Joining_Designation</th>
      <th>Designation</th>
      <th>Minimum_Quarterly_Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
      <td>Master</td>
      <td>65603</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
      <td>College</td>
      <td>46368</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
      <td>Bachelor</td>
      <td>78728</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>24</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 10 columns</p>
</div>




```python
qua_rat = train.groupby(['Emp_ID'])['Quarterly Rating'].agg(['max'])
qua_rat = qua_rat.reset_index()
qua_rat.columns = ['Emp_ID','Maximum_Quarterly_Rating']
qua_rat

def get_rat(x):
  return qua_rat[qua_rat.Emp_ID==x].Maximum_Quarterly_Rating.iloc[0]

df["Maximum_Quarterly_Rating"] = qua_rat.Emp_ID.apply(get_rat)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
      <th>months_worked</th>
      <th>Joining_Designation</th>
      <th>Designation</th>
      <th>Minimum_Quarterly_Rating</th>
      <th>Maximum_Quarterly_Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
      <td>Master</td>
      <td>65603</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
      <td>College</td>
      <td>46368</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
      <td>Bachelor</td>
      <td>78728</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>24</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 11 columns</p>
</div>




```python
doj = train.groupby(['Emp_ID','Dateofjoining']).Dateofjoining.agg('count').to_frame()
doj.columns = ['doj_count']
doj = doj.reset_index()

def get_doj(x):
  return doj[doj.Emp_ID==x].Dateofjoining.iloc[0]

df["Dateofjoining"] = doj.Emp_ID.apply(get_doj)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emp_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>City</th>
      <th>Education_Level</th>
      <th>Salary</th>
      <th>months_worked</th>
      <th>Joining_Designation</th>
      <th>Designation</th>
      <th>Minimum_Quarterly_Rating</th>
      <th>Maximum_Quarterly_Rating</th>
      <th>Dateofjoining</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28</td>
      <td>Male</td>
      <td>C23</td>
      <td>Master</td>
      <td>57387</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2015-12-24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>31</td>
      <td>Male</td>
      <td>C7</td>
      <td>Master</td>
      <td>67016</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2017-11-06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>43</td>
      <td>Male</td>
      <td>C13</td>
      <td>Master</td>
      <td>65603</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2016-12-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>29</td>
      <td>Male</td>
      <td>C9</td>
      <td>College</td>
      <td>46368</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2016-01-09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>31</td>
      <td>Female</td>
      <td>C11</td>
      <td>Bachelor</td>
      <td>78728</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2017-07-31</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>24</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2012-10-15</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2017-08-28</td>
    </tr>
    <tr>
      <th>2378</th>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2015-07-31</td>
    </tr>
    <tr>
      <th>2379</th>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2015-07-21</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2017-06-08</td>
    </tr>
  </tbody>
</table>
<p>2381 rows × 12 columns</p>
</div>




```python

```


```python
X_train,X_val, y_train,y_val = train_test_split(X,y,random_state=42,test_size=0.2)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-17-d1e2aab8c015> in <module>
    ----> 1 X_train,X_val, y_train,y_val = train_test_split(X,y,random_state=42,test_size=0.2)
    

    NameError: name 'X' is not defined



```python
model = CatBoostRegressor(loss_function='RMSE')
model.fit(X_train,y_train,cat_features=['Gender','City','Education_Level','Dateofjoining'])
```


```python
y_pred_val = model.predict(X_val)
```


```python
r2_score(y_pred_val,y_val)
```


```python
# df = df.drop('Dateofjoining',axis=1)

```


```python
df
X = df.drop('months_worked',axis=1)
y = df['months_worked']
```


```python
model.fit(X,y,cat_features=['Gender','City','Education_Level'])
```


```python
merged_right = pd.merge(left=df, right=test, how='right', left_on='Emp_ID', right_on='Emp_ID')
merged_right
```


```python
y_pred = model.predict(merged_right)
```


```python
mysub = pd.read_csv('sample_submission_znWiLZ4.csv')
```


```python
mysub.Target=y_pred
```


```python
mysub.to_csv('mysub.csv',index=False)
```


```python

```
