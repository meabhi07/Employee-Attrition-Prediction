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
train.tail(40)
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
      <th>19064</th>
      <td>2016-10-01</td>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>990000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19065</th>
      <td>2016-11-01</td>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>591710</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19066</th>
      <td>2016-12-01</td>
      <td>2784</td>
      <td>33</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>194010</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19067</th>
      <td>2017-01-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>1309620</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19068</th>
      <td>2017-02-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>850050</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19069</th>
      <td>2017-03-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>4128460</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19070</th>
      <td>2017-04-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>150260</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19071</th>
      <td>2017-05-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>153800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19072</th>
      <td>2017-06-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>979270</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19073</th>
      <td>2017-07-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>252000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19074</th>
      <td>2017-08-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>1260090</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19075</th>
      <td>2017-09-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>400000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19076</th>
      <td>2017-10-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>3087830</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19077</th>
      <td>2017-11-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19078</th>
      <td>2017-12-01</td>
      <td>2784</td>
      <td>34</td>
      <td>Male</td>
      <td>C24</td>
      <td>College</td>
      <td>82815</td>
      <td>2012-10-15</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>505480</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19079</th>
      <td>2017-08-01</td>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>2017-08-28</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19080</th>
      <td>2017-09-01</td>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>2017-08-28</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19081</th>
      <td>2017-10-01</td>
      <td>2785</td>
      <td>34</td>
      <td>Female</td>
      <td>C9</td>
      <td>College</td>
      <td>12105</td>
      <td>2017-08-28</td>
      <td>2017-10-28</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19082</th>
      <td>2016-01-01</td>
      <td>2786</td>
      <td>44</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>221080</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19083</th>
      <td>2016-02-01</td>
      <td>2786</td>
      <td>45</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>485270</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19084</th>
      <td>2016-03-01</td>
      <td>2786</td>
      <td>45</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>970380</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19085</th>
      <td>2016-04-01</td>
      <td>2786</td>
      <td>45</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>432240</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19086</th>
      <td>2016-05-01</td>
      <td>2786</td>
      <td>45</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>387660</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19087</th>
      <td>2016-06-01</td>
      <td>2786</td>
      <td>45</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19088</th>
      <td>2016-07-01</td>
      <td>2786</td>
      <td>45</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>318460</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19089</th>
      <td>2016-08-01</td>
      <td>2786</td>
      <td>45</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19090</th>
      <td>2016-09-01</td>
      <td>2786</td>
      <td>45</td>
      <td>Male</td>
      <td>C19</td>
      <td>College</td>
      <td>35370</td>
      <td>2015-07-31</td>
      <td>2016-09-22</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19091</th>
      <td>2016-01-01</td>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>2015-07-21</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>408090</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19092</th>
      <td>2016-02-01</td>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>2015-07-21</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>250000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19093</th>
      <td>2016-03-01</td>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>2015-07-21</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>319740</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19094</th>
      <td>2016-04-01</td>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>2015-07-21</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19095</th>
      <td>2016-05-01</td>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>2015-07-21</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19096</th>
      <td>2016-06-01</td>
      <td>2787</td>
      <td>28</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>69498</td>
      <td>2015-07-21</td>
      <td>2016-06-20</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19097</th>
      <td>2017-06-01</td>
      <td>2788</td>
      <td>29</td>
      <td>Male</td>
      <td>C27</td>
      <td>Master</td>
      <td>70254</td>
      <td>2017-06-08</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19098</th>
      <td>2017-07-01</td>
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
      <td>497690</td>
      <td>3</td>
    </tr>
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


```python
model = CatBoostRegressor(loss_function='RMSE')
model.fit(X_train,y_train,cat_features=['Gender','City','Education_Level','Dateofjoining'])
```

    Learning rate set to 0.04377
    0:	learn: 6.6102314	total: 12.4ms	remaining: 12.4s
    1:	learn: 6.4288079	total: 40ms	remaining: 20s
    2:	learn: 6.2719022	total: 51.5ms	remaining: 17.1s
    3:	learn: 6.1260937	total: 62.5ms	remaining: 15.6s
    4:	learn: 6.0025147	total: 70.3ms	remaining: 14s
    5:	learn: 5.8598481	total: 81.1ms	remaining: 13.4s
    6:	learn: 5.7602189	total: 85ms	remaining: 12.1s
    7:	learn: 5.6325447	total: 103ms	remaining: 12.8s
    8:	learn: 5.5329750	total: 114ms	remaining: 12.6s
    9:	learn: 5.4316741	total: 122ms	remaining: 12.1s
    10:	learn: 5.3339449	total: 131ms	remaining: 11.7s
    11:	learn: 5.2340067	total: 142ms	remaining: 11.7s
    12:	learn: 5.1343744	total: 154ms	remaining: 11.7s
    13:	learn: 5.0428565	total: 165ms	remaining: 11.6s
    14:	learn: 4.9567524	total: 177ms	remaining: 11.6s
    15:	learn: 4.8806230	total: 186ms	remaining: 11.4s
    16:	learn: 4.8222398	total: 192ms	remaining: 11.1s
    17:	learn: 4.7577971	total: 201ms	remaining: 11s
    18:	learn: 4.6905396	total: 213ms	remaining: 11s
    19:	learn: 4.6316541	total: 224ms	remaining: 11s
    20:	learn: 4.5803203	total: 236ms	remaining: 11s
    21:	learn: 4.5242507	total: 247ms	remaining: 11s
    22:	learn: 4.4633981	total: 258ms	remaining: 11s
    23:	learn: 4.4157142	total: 286ms	remaining: 11.6s
    24:	learn: 4.3631157	total: 299ms	remaining: 11.7s
    25:	learn: 4.3225316	total: 311ms	remaining: 11.6s
    26:	learn: 4.2880693	total: 322ms	remaining: 11.6s
    27:	learn: 4.2506030	total: 334ms	remaining: 11.6s
    28:	learn: 4.2117621	total: 345ms	remaining: 11.6s
    29:	learn: 4.1707948	total: 357ms	remaining: 11.5s
    30:	learn: 4.1364066	total: 378ms	remaining: 11.8s
    31:	learn: 4.1043682	total: 392ms	remaining: 11.9s
    32:	learn: 4.0748964	total: 403ms	remaining: 11.8s
    33:	learn: 4.0408666	total: 415ms	remaining: 11.8s
    34:	learn: 4.0104642	total: 427ms	remaining: 11.8s
    35:	learn: 3.9793678	total: 438ms	remaining: 11.7s
    36:	learn: 3.9534630	total: 449ms	remaining: 11.7s
    37:	learn: 3.9321234	total: 461ms	remaining: 11.7s
    38:	learn: 3.9219932	total: 464ms	remaining: 11.4s
    39:	learn: 3.8994659	total: 488ms	remaining: 11.7s
    40:	learn: 3.8859087	total: 496ms	remaining: 11.6s
    41:	learn: 3.8659998	total: 508ms	remaining: 11.6s
    42:	learn: 3.8435962	total: 520ms	remaining: 11.6s
    43:	learn: 3.8244792	total: 534ms	remaining: 11.6s
    44:	learn: 3.8047154	total: 546ms	remaining: 11.6s
    45:	learn: 3.7931649	total: 553ms	remaining: 11.5s
    46:	learn: 3.7765018	total: 562ms	remaining: 11.4s
    47:	learn: 3.7630326	total: 573ms	remaining: 11.4s
    48:	learn: 3.7509304	total: 582ms	remaining: 11.3s
    49:	learn: 3.7391227	total: 594ms	remaining: 11.3s
    50:	learn: 3.7267535	total: 605ms	remaining: 11.3s
    51:	learn: 3.7175824	total: 613ms	remaining: 11.2s
    52:	learn: 3.7062639	total: 624ms	remaining: 11.1s
    53:	learn: 3.6983353	total: 635ms	remaining: 11.1s
    54:	learn: 3.6865714	total: 646ms	remaining: 11.1s
    55:	learn: 3.6757579	total: 658ms	remaining: 11.1s
    56:	learn: 3.6658768	total: 669ms	remaining: 11.1s
    57:	learn: 3.6499351	total: 680ms	remaining: 11s
    58:	learn: 3.6381569	total: 691ms	remaining: 11s
    59:	learn: 3.6222291	total: 704ms	remaining: 11s
    60:	learn: 3.6112409	total: 716ms	remaining: 11s
    61:	learn: 3.5968048	total: 727ms	remaining: 11s
    62:	learn: 3.5899821	total: 741ms	remaining: 11s
    63:	learn: 3.5853030	total: 754ms	remaining: 11s
    64:	learn: 3.5807055	total: 759ms	remaining: 10.9s
    65:	learn: 3.5745071	total: 770ms	remaining: 10.9s
    66:	learn: 3.5655324	total: 783ms	remaining: 10.9s
    67:	learn: 3.5596386	total: 794ms	remaining: 10.9s
    68:	learn: 3.5508414	total: 805ms	remaining: 10.9s
    69:	learn: 3.5471504	total: 817ms	remaining: 10.8s
    70:	learn: 3.5397489	total: 829ms	remaining: 10.8s
    71:	learn: 3.5314089	total: 843ms	remaining: 10.9s
    72:	learn: 3.5198002	total: 856ms	remaining: 10.9s
    73:	learn: 3.5139962	total: 867ms	remaining: 10.9s
    74:	learn: 3.5091060	total: 881ms	remaining: 10.9s
    75:	learn: 3.5044499	total: 908ms	remaining: 11s
    76:	learn: 3.4992626	total: 921ms	remaining: 11s
    77:	learn: 3.4981494	total: 927ms	remaining: 11s
    78:	learn: 3.4902829	total: 940ms	remaining: 11s
    79:	learn: 3.4862421	total: 952ms	remaining: 10.9s
    80:	learn: 3.4850255	total: 958ms	remaining: 10.9s
    81:	learn: 3.4816251	total: 985ms	remaining: 11s
    82:	learn: 3.4793401	total: 1s	remaining: 11.1s
    83:	learn: 3.4780289	total: 1.01s	remaining: 11s
    84:	learn: 3.4710456	total: 1.02s	remaining: 11s
    85:	learn: 3.4642778	total: 1.04s	remaining: 11.1s
    86:	learn: 3.4621448	total: 1.06s	remaining: 11.1s
    87:	learn: 3.4598135	total: 1.07s	remaining: 11.1s
    88:	learn: 3.4584143	total: 1.08s	remaining: 11.1s
    89:	learn: 3.4549160	total: 1.09s	remaining: 11s
    90:	learn: 3.4530950	total: 1.1s	remaining: 11s
    91:	learn: 3.4516415	total: 1.11s	remaining: 11s
    92:	learn: 3.4495184	total: 1.13s	remaining: 11s
    93:	learn: 3.4450539	total: 1.14s	remaining: 11s
    94:	learn: 3.4425329	total: 1.15s	remaining: 11s
    95:	learn: 3.4403879	total: 1.16s	remaining: 11s
    96:	learn: 3.4342204	total: 1.17s	remaining: 10.9s
    97:	learn: 3.4325493	total: 1.19s	remaining: 10.9s
    98:	learn: 3.4286376	total: 1.2s	remaining: 10.9s
    99:	learn: 3.4260448	total: 1.21s	remaining: 10.9s
    100:	learn: 3.4246891	total: 1.22s	remaining: 10.9s
    101:	learn: 3.4200926	total: 1.23s	remaining: 10.9s
    102:	learn: 3.4144702	total: 1.25s	remaining: 10.9s
    103:	learn: 3.4113753	total: 1.26s	remaining: 10.8s
    104:	learn: 3.4091321	total: 1.27s	remaining: 10.9s
    105:	learn: 3.4036920	total: 1.29s	remaining: 10.9s
    106:	learn: 3.4035623	total: 1.29s	remaining: 10.8s
    107:	learn: 3.4000354	total: 1.32s	remaining: 10.9s
    108:	learn: 3.3972140	total: 1.34s	remaining: 10.9s
    109:	learn: 3.3945449	total: 1.35s	remaining: 10.9s
    110:	learn: 3.3924167	total: 1.37s	remaining: 10.9s
    111:	learn: 3.3890586	total: 1.38s	remaining: 10.9s
    112:	learn: 3.3841470	total: 1.39s	remaining: 10.9s
    113:	learn: 3.3801740	total: 1.41s	remaining: 10.9s
    114:	learn: 3.3758763	total: 1.42s	remaining: 10.9s
    115:	learn: 3.3727479	total: 1.46s	remaining: 11.1s
    116:	learn: 3.3689535	total: 1.47s	remaining: 11.1s
    117:	learn: 3.3664846	total: 1.48s	remaining: 11.1s
    118:	learn: 3.3624417	total: 1.49s	remaining: 11.1s
    119:	learn: 3.3612639	total: 1.51s	remaining: 11.1s
    120:	learn: 3.3597220	total: 1.53s	remaining: 11.1s
    121:	learn: 3.3588468	total: 1.54s	remaining: 11.1s
    122:	learn: 3.3563843	total: 1.55s	remaining: 11s
    123:	learn: 3.3534902	total: 1.56s	remaining: 11s
    124:	learn: 3.3491531	total: 1.57s	remaining: 11s
    125:	learn: 3.3476315	total: 1.58s	remaining: 11s
    126:	learn: 3.3446843	total: 1.6s	remaining: 11s
    127:	learn: 3.3439793	total: 1.61s	remaining: 11s
    128:	learn: 3.3400999	total: 1.62s	remaining: 10.9s
    129:	learn: 3.3359739	total: 1.63s	remaining: 10.9s
    130:	learn: 3.3350702	total: 1.64s	remaining: 10.9s
    131:	learn: 3.3323082	total: 1.65s	remaining: 10.9s
    132:	learn: 3.3318111	total: 1.66s	remaining: 10.8s
    133:	learn: 3.3302704	total: 1.67s	remaining: 10.8s
    134:	learn: 3.3262412	total: 1.68s	remaining: 10.8s
    135:	learn: 3.3253777	total: 1.69s	remaining: 10.7s
    136:	learn: 3.3233454	total: 1.7s	remaining: 10.7s
    137:	learn: 3.3218732	total: 1.71s	remaining: 10.7s
    138:	learn: 3.3194780	total: 1.73s	remaining: 10.7s
    139:	learn: 3.3170496	total: 1.74s	remaining: 10.7s
    140:	learn: 3.3159938	total: 1.75s	remaining: 10.6s
    141:	learn: 3.3148153	total: 1.76s	remaining: 10.6s
    142:	learn: 3.3133115	total: 1.77s	remaining: 10.6s
    143:	learn: 3.3125961	total: 1.78s	remaining: 10.6s
    144:	learn: 3.3112129	total: 1.79s	remaining: 10.6s
    145:	learn: 3.3096586	total: 1.8s	remaining: 10.5s
    146:	learn: 3.3052225	total: 1.81s	remaining: 10.5s
    147:	learn: 3.3022974	total: 1.82s	remaining: 10.5s
    148:	learn: 3.3011851	total: 1.83s	remaining: 10.5s
    149:	learn: 3.2996873	total: 1.85s	remaining: 10.5s
    150:	learn: 3.2982650	total: 1.86s	remaining: 10.4s
    151:	learn: 3.2958507	total: 1.87s	remaining: 10.4s
    152:	learn: 3.2944164	total: 1.88s	remaining: 10.4s
    153:	learn: 3.2930985	total: 1.9s	remaining: 10.4s
    154:	learn: 3.2901820	total: 1.91s	remaining: 10.4s
    155:	learn: 3.2888220	total: 1.92s	remaining: 10.4s
    156:	learn: 3.2877015	total: 1.93s	remaining: 10.4s
    157:	learn: 3.2848352	total: 1.94s	remaining: 10.3s
    158:	learn: 3.2825143	total: 1.95s	remaining: 10.3s
    159:	learn: 3.2806680	total: 1.97s	remaining: 10.3s
    160:	learn: 3.2796469	total: 1.98s	remaining: 10.3s
    161:	learn: 3.2789703	total: 1.99s	remaining: 10.3s
    162:	learn: 3.2774331	total: 2s	remaining: 10.3s
    163:	learn: 3.2752154	total: 2.01s	remaining: 10.3s
    164:	learn: 3.2734884	total: 2.03s	remaining: 10.3s
    165:	learn: 3.2715465	total: 2.04s	remaining: 10.2s
    166:	learn: 3.2671765	total: 2.05s	remaining: 10.2s
    167:	learn: 3.2655651	total: 2.08s	remaining: 10.3s
    168:	learn: 3.2643869	total: 2.09s	remaining: 10.3s
    169:	learn: 3.2633759	total: 2.1s	remaining: 10.3s
    170:	learn: 3.2633694	total: 2.11s	remaining: 10.2s
    171:	learn: 3.2625635	total: 2.12s	remaining: 10.2s
    172:	learn: 3.2613617	total: 2.13s	remaining: 10.2s
    173:	learn: 3.2570438	total: 2.14s	remaining: 10.2s
    174:	learn: 3.2550176	total: 2.17s	remaining: 10.2s
    175:	learn: 3.2534695	total: 2.18s	remaining: 10.2s
    176:	learn: 3.2516093	total: 2.2s	remaining: 10.2s
    177:	learn: 3.2493483	total: 2.21s	remaining: 10.2s
    178:	learn: 3.2452587	total: 2.22s	remaining: 10.2s
    179:	learn: 3.2434957	total: 2.23s	remaining: 10.2s
    180:	learn: 3.2418541	total: 2.25s	remaining: 10.2s
    181:	learn: 3.2403467	total: 2.26s	remaining: 10.1s
    182:	learn: 3.2386705	total: 2.27s	remaining: 10.1s
    183:	learn: 3.2374927	total: 2.29s	remaining: 10.1s
    184:	learn: 3.2347276	total: 2.3s	remaining: 10.1s
    185:	learn: 3.2305954	total: 2.31s	remaining: 10.1s
    186:	learn: 3.2290858	total: 2.32s	remaining: 10.1s
    187:	learn: 3.2268933	total: 2.34s	remaining: 10.1s
    188:	learn: 3.2243429	total: 2.35s	remaining: 10.1s
    189:	learn: 3.2232739	total: 2.37s	remaining: 10.1s
    190:	learn: 3.2224124	total: 2.38s	remaining: 10.1s
    191:	learn: 3.2198510	total: 2.39s	remaining: 10.1s
    192:	learn: 3.2160084	total: 2.4s	remaining: 10.1s
    193:	learn: 3.2153983	total: 2.42s	remaining: 10s
    194:	learn: 3.2136000	total: 2.43s	remaining: 10s
    195:	learn: 3.2113054	total: 2.44s	remaining: 10s
    196:	learn: 3.2096336	total: 2.45s	remaining: 10s
    197:	learn: 3.2074712	total: 2.47s	remaining: 9.99s
    198:	learn: 3.2061395	total: 2.48s	remaining: 9.98s
    199:	learn: 3.2059452	total: 2.49s	remaining: 9.97s
    200:	learn: 3.2037070	total: 2.51s	remaining: 9.96s
    201:	learn: 3.2016890	total: 2.52s	remaining: 9.96s
    202:	learn: 3.1995182	total: 2.55s	remaining: 10s
    203:	learn: 3.1985330	total: 2.56s	remaining: 10s
    204:	learn: 3.1961221	total: 2.58s	remaining: 9.99s
    205:	learn: 3.1937372	total: 2.59s	remaining: 9.98s
    206:	learn: 3.1922050	total: 2.6s	remaining: 9.96s
    207:	learn: 3.1900804	total: 2.61s	remaining: 9.95s
    208:	learn: 3.1894789	total: 2.62s	remaining: 9.93s
    209:	learn: 3.1888268	total: 2.63s	remaining: 9.91s
    210:	learn: 3.1868036	total: 2.65s	remaining: 9.89s
    211:	learn: 3.1861069	total: 2.66s	remaining: 9.87s
    212:	learn: 3.1845194	total: 2.67s	remaining: 9.87s
    213:	learn: 3.1819997	total: 2.68s	remaining: 9.85s
    214:	learn: 3.1803450	total: 2.69s	remaining: 9.83s
    215:	learn: 3.1793627	total: 2.7s	remaining: 9.81s
    216:	learn: 3.1785069	total: 2.71s	remaining: 9.8s
    217:	learn: 3.1774984	total: 2.73s	remaining: 9.78s
    218:	learn: 3.1753889	total: 2.74s	remaining: 9.76s
    219:	learn: 3.1733880	total: 2.75s	remaining: 9.74s
    220:	learn: 3.1727456	total: 2.76s	remaining: 9.72s
    221:	learn: 3.1722048	total: 2.77s	remaining: 9.71s
    222:	learn: 3.1708294	total: 2.78s	remaining: 9.7s
    223:	learn: 3.1685579	total: 2.8s	remaining: 9.69s
    224:	learn: 3.1668548	total: 2.81s	remaining: 9.67s
    225:	learn: 3.1657004	total: 2.82s	remaining: 9.66s
    226:	learn: 3.1644799	total: 2.83s	remaining: 9.64s
    227:	learn: 3.1625554	total: 2.84s	remaining: 9.62s
    228:	learn: 3.1613258	total: 2.85s	remaining: 9.61s
    229:	learn: 3.1593305	total: 2.86s	remaining: 9.59s
    230:	learn: 3.1577890	total: 2.89s	remaining: 9.61s
    231:	learn: 3.1546834	total: 2.9s	remaining: 9.6s
    232:	learn: 3.1534601	total: 2.91s	remaining: 9.59s
    233:	learn: 3.1527726	total: 2.92s	remaining: 9.57s
    234:	learn: 3.1506957	total: 2.94s	remaining: 9.56s
    235:	learn: 3.1480104	total: 2.95s	remaining: 9.54s
    236:	learn: 3.1468900	total: 2.96s	remaining: 9.52s
    237:	learn: 3.1454500	total: 2.97s	remaining: 9.51s
    238:	learn: 3.1448196	total: 2.99s	remaining: 9.51s
    239:	learn: 3.1427102	total: 3s	remaining: 9.49s
    240:	learn: 3.1397021	total: 3.01s	remaining: 9.48s
    241:	learn: 3.1364410	total: 3.02s	remaining: 9.46s
    242:	learn: 3.1344484	total: 3.03s	remaining: 9.45s
    243:	learn: 3.1329278	total: 3.04s	remaining: 9.43s
    244:	learn: 3.1316817	total: 3.06s	remaining: 9.42s
    245:	learn: 3.1298096	total: 3.07s	remaining: 9.41s
    246:	learn: 3.1285312	total: 3.08s	remaining: 9.39s
    247:	learn: 3.1271618	total: 3.09s	remaining: 9.38s
    248:	learn: 3.1249403	total: 3.1s	remaining: 9.37s
    249:	learn: 3.1228012	total: 3.12s	remaining: 9.36s
    250:	learn: 3.1202155	total: 3.13s	remaining: 9.34s
    251:	learn: 3.1183658	total: 3.16s	remaining: 9.37s
    252:	learn: 3.1166865	total: 3.17s	remaining: 9.36s
    253:	learn: 3.1153112	total: 3.18s	remaining: 9.34s
    254:	learn: 3.1144802	total: 3.2s	remaining: 9.35s
    255:	learn: 3.1127742	total: 3.21s	remaining: 9.33s
    256:	learn: 3.1094618	total: 3.22s	remaining: 9.32s
    257:	learn: 3.1060073	total: 3.23s	remaining: 9.3s
    258:	learn: 3.1042232	total: 3.25s	remaining: 9.29s
    259:	learn: 3.1025546	total: 3.26s	remaining: 9.28s
    260:	learn: 3.1001139	total: 3.27s	remaining: 9.26s
    261:	learn: 3.0977643	total: 3.28s	remaining: 9.25s
    262:	learn: 3.0965908	total: 3.29s	remaining: 9.23s
    263:	learn: 3.0954604	total: 3.32s	remaining: 9.25s
    264:	learn: 3.0945005	total: 3.33s	remaining: 9.24s
    265:	learn: 3.0902249	total: 3.34s	remaining: 9.22s
    266:	learn: 3.0877364	total: 3.35s	remaining: 9.21s
    267:	learn: 3.0874406	total: 3.36s	remaining: 9.19s
    268:	learn: 3.0858761	total: 3.38s	remaining: 9.17s
    269:	learn: 3.0847317	total: 3.39s	remaining: 9.16s
    270:	learn: 3.0837530	total: 3.4s	remaining: 9.15s
    271:	learn: 3.0823056	total: 3.41s	remaining: 9.14s
    272:	learn: 3.0798494	total: 3.42s	remaining: 9.12s
    273:	learn: 3.0784887	total: 3.44s	remaining: 9.11s
    274:	learn: 3.0759899	total: 3.45s	remaining: 9.09s
    275:	learn: 3.0754177	total: 3.46s	remaining: 9.07s
    276:	learn: 3.0745221	total: 3.47s	remaining: 9.06s
    277:	learn: 3.0725600	total: 3.49s	remaining: 9.07s
    278:	learn: 3.0688519	total: 3.51s	remaining: 9.06s
    279:	learn: 3.0672473	total: 3.53s	remaining: 9.07s
    280:	learn: 3.0670983	total: 3.54s	remaining: 9.05s
    281:	learn: 3.0652593	total: 3.55s	remaining: 9.03s
    282:	learn: 3.0619034	total: 3.56s	remaining: 9.02s
    283:	learn: 3.0595852	total: 3.57s	remaining: 9s
    284:	learn: 3.0573999	total: 3.58s	remaining: 8.99s
    285:	learn: 3.0554504	total: 3.6s	remaining: 8.98s
    286:	learn: 3.0548115	total: 3.61s	remaining: 8.96s
    287:	learn: 3.0534855	total: 3.62s	remaining: 8.95s
    288:	learn: 3.0524289	total: 3.63s	remaining: 8.94s
    289:	learn: 3.0519329	total: 3.64s	remaining: 8.92s
    290:	learn: 3.0506395	total: 3.65s	remaining: 8.9s
    291:	learn: 3.0502998	total: 3.67s	remaining: 8.89s
    292:	learn: 3.0500326	total: 3.68s	remaining: 8.87s
    293:	learn: 3.0476339	total: 3.69s	remaining: 8.86s
    294:	learn: 3.0455508	total: 3.7s	remaining: 8.85s
    295:	learn: 3.0445264	total: 3.73s	remaining: 8.86s
    296:	learn: 3.0424198	total: 3.74s	remaining: 8.85s
    297:	learn: 3.0406966	total: 3.75s	remaining: 8.83s
    298:	learn: 3.0385102	total: 3.76s	remaining: 8.81s
    299:	learn: 3.0360143	total: 3.77s	remaining: 8.8s
    300:	learn: 3.0351709	total: 3.78s	remaining: 8.79s
    301:	learn: 3.0329802	total: 3.8s	remaining: 8.77s
    302:	learn: 3.0315949	total: 3.81s	remaining: 8.76s
    303:	learn: 3.0301122	total: 3.82s	remaining: 8.74s
    304:	learn: 3.0268189	total: 3.83s	remaining: 8.73s
    305:	learn: 3.0242808	total: 3.84s	remaining: 8.71s
    306:	learn: 3.0216699	total: 3.85s	remaining: 8.7s
    307:	learn: 3.0212729	total: 3.87s	remaining: 8.69s
    308:	learn: 3.0194911	total: 3.88s	remaining: 8.67s
    309:	learn: 3.0186501	total: 3.89s	remaining: 8.66s
    310:	learn: 3.0180855	total: 3.9s	remaining: 8.64s
    311:	learn: 3.0157344	total: 3.92s	remaining: 8.63s
    312:	learn: 3.0142927	total: 3.93s	remaining: 8.62s
    313:	learn: 3.0108553	total: 3.94s	remaining: 8.6s
    314:	learn: 3.0077899	total: 3.95s	remaining: 8.59s
    315:	learn: 3.0064903	total: 3.97s	remaining: 8.6s
    316:	learn: 3.0050790	total: 3.98s	remaining: 8.59s
    317:	learn: 3.0041220	total: 4s	remaining: 8.57s
    318:	learn: 3.0032322	total: 4.01s	remaining: 8.55s
    319:	learn: 3.0019825	total: 4.02s	remaining: 8.54s
    320:	learn: 2.9992791	total: 4.03s	remaining: 8.53s
    321:	learn: 2.9977448	total: 4.05s	remaining: 8.53s
    322:	learn: 2.9971699	total: 4.06s	remaining: 8.51s
    323:	learn: 2.9965587	total: 4.07s	remaining: 8.5s
    324:	learn: 2.9960709	total: 4.11s	remaining: 8.54s
    325:	learn: 2.9949670	total: 4.12s	remaining: 8.53s
    326:	learn: 2.9940282	total: 4.13s	remaining: 8.51s
    327:	learn: 2.9896833	total: 4.15s	remaining: 8.49s
    328:	learn: 2.9879147	total: 4.16s	remaining: 8.48s
    329:	learn: 2.9872674	total: 4.17s	remaining: 8.46s
    330:	learn: 2.9863224	total: 4.18s	remaining: 8.45s
    331:	learn: 2.9836537	total: 4.19s	remaining: 8.44s
    332:	learn: 2.9821047	total: 4.24s	remaining: 8.48s
    333:	learn: 2.9804611	total: 4.25s	remaining: 8.47s
    334:	learn: 2.9784255	total: 4.26s	remaining: 8.46s
    335:	learn: 2.9768597	total: 4.27s	remaining: 8.45s
    336:	learn: 2.9763427	total: 4.3s	remaining: 8.45s
    337:	learn: 2.9731091	total: 4.31s	remaining: 8.44s
    338:	learn: 2.9726340	total: 4.33s	remaining: 8.44s
    339:	learn: 2.9711258	total: 4.34s	remaining: 8.43s
    340:	learn: 2.9680776	total: 4.35s	remaining: 8.41s
    341:	learn: 2.9665951	total: 4.37s	remaining: 8.4s
    342:	learn: 2.9653547	total: 4.38s	remaining: 8.39s
    343:	learn: 2.9620944	total: 4.39s	remaining: 8.38s
    344:	learn: 2.9595968	total: 4.42s	remaining: 8.39s
    345:	learn: 2.9584850	total: 4.43s	remaining: 8.37s
    346:	learn: 2.9580287	total: 4.44s	remaining: 8.36s
    347:	learn: 2.9569900	total: 4.46s	remaining: 8.36s
    348:	learn: 2.9564400	total: 4.47s	remaining: 8.34s
    349:	learn: 2.9536923	total: 4.48s	remaining: 8.33s
    350:	learn: 2.9528249	total: 4.5s	remaining: 8.31s
    351:	learn: 2.9503627	total: 4.51s	remaining: 8.3s
    352:	learn: 2.9495194	total: 4.53s	remaining: 8.29s
    353:	learn: 2.9465165	total: 4.54s	remaining: 8.29s
    354:	learn: 2.9442991	total: 4.55s	remaining: 8.27s
    355:	learn: 2.9434162	total: 4.56s	remaining: 8.26s
    356:	learn: 2.9432501	total: 4.58s	remaining: 8.24s
    357:	learn: 2.9425363	total: 4.59s	remaining: 8.23s
    358:	learn: 2.9414373	total: 4.6s	remaining: 8.21s
    359:	learn: 2.9410088	total: 4.61s	remaining: 8.2s
    360:	learn: 2.9397246	total: 4.62s	remaining: 8.18s
    361:	learn: 2.9390810	total: 4.63s	remaining: 8.17s
    362:	learn: 2.9386777	total: 4.64s	remaining: 8.15s
    363:	learn: 2.9385098	total: 4.66s	remaining: 8.13s
    364:	learn: 2.9362116	total: 4.67s	remaining: 8.12s
    365:	learn: 2.9346134	total: 4.68s	remaining: 8.11s
    366:	learn: 2.9319914	total: 4.69s	remaining: 8.09s
    367:	learn: 2.9293781	total: 4.7s	remaining: 8.07s
    368:	learn: 2.9280857	total: 4.71s	remaining: 8.06s
    369:	learn: 2.9273601	total: 4.73s	remaining: 8.05s
    370:	learn: 2.9256976	total: 4.74s	remaining: 8.03s
    371:	learn: 2.9254517	total: 4.77s	remaining: 8.06s
    372:	learn: 2.9250003	total: 4.78s	remaining: 8.04s
    373:	learn: 2.9239056	total: 4.8s	remaining: 8.03s
    374:	learn: 2.9230656	total: 4.81s	remaining: 8.02s
    375:	learn: 2.9229116	total: 4.82s	remaining: 8s
    376:	learn: 2.9215046	total: 4.83s	remaining: 7.98s
    377:	learn: 2.9199455	total: 4.84s	remaining: 7.97s
    378:	learn: 2.9192943	total: 4.87s	remaining: 7.98s
    379:	learn: 2.9163870	total: 4.88s	remaining: 7.96s
    380:	learn: 2.9145244	total: 4.89s	remaining: 7.95s
    381:	learn: 2.9121987	total: 4.91s	remaining: 7.93s
    382:	learn: 2.9097525	total: 4.92s	remaining: 7.92s
    383:	learn: 2.9074593	total: 4.93s	remaining: 7.91s
    384:	learn: 2.9068313	total: 4.94s	remaining: 7.89s
    385:	learn: 2.9063106	total: 4.95s	remaining: 7.88s
    386:	learn: 2.9049555	total: 4.96s	remaining: 7.86s
    387:	learn: 2.9044844	total: 4.97s	remaining: 7.84s
    388:	learn: 2.9037254	total: 4.98s	remaining: 7.83s
    389:	learn: 2.9015810	total: 5s	remaining: 7.82s
    390:	learn: 2.9001960	total: 5.01s	remaining: 7.8s
    391:	learn: 2.8982458	total: 5.02s	remaining: 7.79s
    392:	learn: 2.8975514	total: 5.03s	remaining: 7.77s
    393:	learn: 2.8962765	total: 5.05s	remaining: 7.77s
    394:	learn: 2.8937878	total: 5.06s	remaining: 7.76s
    395:	learn: 2.8923709	total: 5.08s	remaining: 7.75s
    396:	learn: 2.8914997	total: 5.09s	remaining: 7.74s
    397:	learn: 2.8905319	total: 5.1s	remaining: 7.72s
    398:	learn: 2.8884706	total: 5.12s	remaining: 7.71s
    399:	learn: 2.8877311	total: 5.13s	remaining: 7.69s
    400:	learn: 2.8872403	total: 5.14s	remaining: 7.68s
    401:	learn: 2.8864851	total: 5.15s	remaining: 7.66s
    402:	learn: 2.8857557	total: 5.16s	remaining: 7.65s
    403:	learn: 2.8826441	total: 5.17s	remaining: 7.63s
    404:	learn: 2.8819433	total: 5.18s	remaining: 7.62s
    405:	learn: 2.8807319	total: 5.2s	remaining: 7.61s
    406:	learn: 2.8794055	total: 5.21s	remaining: 7.59s
    407:	learn: 2.8770218	total: 5.22s	remaining: 7.57s
    408:	learn: 2.8756756	total: 5.23s	remaining: 7.56s
    409:	learn: 2.8739107	total: 5.24s	remaining: 7.54s
    410:	learn: 2.8735706	total: 5.25s	remaining: 7.53s
    411:	learn: 2.8734029	total: 5.26s	remaining: 7.51s
    412:	learn: 2.8701415	total: 5.28s	remaining: 7.5s
    413:	learn: 2.8686336	total: 5.29s	remaining: 7.48s
    414:	learn: 2.8679194	total: 5.3s	remaining: 7.47s
    415:	learn: 2.8667018	total: 5.31s	remaining: 7.46s
    416:	learn: 2.8658395	total: 5.32s	remaining: 7.44s
    417:	learn: 2.8635652	total: 5.33s	remaining: 7.43s
    418:	learn: 2.8621405	total: 5.34s	remaining: 7.41s
    419:	learn: 2.8613875	total: 5.36s	remaining: 7.4s
    420:	learn: 2.8600987	total: 5.37s	remaining: 7.38s
    421:	learn: 2.8581230	total: 5.38s	remaining: 7.37s
    422:	learn: 2.8568424	total: 5.39s	remaining: 7.36s
    423:	learn: 2.8557622	total: 5.41s	remaining: 7.34s
    424:	learn: 2.8530893	total: 5.42s	remaining: 7.33s
    425:	learn: 2.8526132	total: 5.43s	remaining: 7.31s
    426:	learn: 2.8514277	total: 5.44s	remaining: 7.3s
    427:	learn: 2.8502158	total: 5.45s	remaining: 7.29s
    428:	learn: 2.8485996	total: 5.46s	remaining: 7.27s
    429:	learn: 2.8471136	total: 5.47s	remaining: 7.26s
    430:	learn: 2.8466240	total: 5.48s	remaining: 7.24s
    431:	learn: 2.8455491	total: 5.5s	remaining: 7.23s
    432:	learn: 2.8451703	total: 5.51s	remaining: 7.21s
    433:	learn: 2.8441386	total: 5.52s	remaining: 7.2s
    434:	learn: 2.8426107	total: 5.53s	remaining: 7.19s
    435:	learn: 2.8413495	total: 5.54s	remaining: 7.17s
    436:	learn: 2.8397883	total: 5.55s	remaining: 7.16s
    437:	learn: 2.8378559	total: 5.57s	remaining: 7.14s
    438:	learn: 2.8359912	total: 5.59s	remaining: 7.14s
    439:	learn: 2.8342771	total: 5.6s	remaining: 7.13s
    440:	learn: 2.8332282	total: 5.62s	remaining: 7.12s
    441:	learn: 2.8316199	total: 5.63s	remaining: 7.1s
    442:	learn: 2.8300219	total: 5.64s	remaining: 7.09s
    443:	learn: 2.8284819	total: 5.65s	remaining: 7.07s
    444:	learn: 2.8277915	total: 5.66s	remaining: 7.06s
    445:	learn: 2.8267469	total: 5.67s	remaining: 7.04s
    446:	learn: 2.8251912	total: 5.68s	remaining: 7.03s
    447:	learn: 2.8239833	total: 5.69s	remaining: 7.01s
    448:	learn: 2.8232418	total: 5.72s	remaining: 7.03s
    449:	learn: 2.8214919	total: 5.74s	remaining: 7.01s
    450:	learn: 2.8201140	total: 5.76s	remaining: 7.01s
    451:	learn: 2.8188007	total: 5.77s	remaining: 7s
    452:	learn: 2.8177553	total: 5.78s	remaining: 6.98s
    453:	learn: 2.8164791	total: 5.8s	remaining: 6.97s
    454:	learn: 2.8151932	total: 5.81s	remaining: 6.96s
    455:	learn: 2.8148569	total: 5.82s	remaining: 6.94s
    456:	learn: 2.8146687	total: 5.83s	remaining: 6.93s
    457:	learn: 2.8133196	total: 5.84s	remaining: 6.91s
    458:	learn: 2.8129228	total: 5.85s	remaining: 6.9s
    459:	learn: 2.8111274	total: 5.86s	remaining: 6.88s
    460:	learn: 2.8093140	total: 5.88s	remaining: 6.87s
    461:	learn: 2.8075925	total: 5.89s	remaining: 6.86s
    462:	learn: 2.8057620	total: 5.9s	remaining: 6.84s
    463:	learn: 2.8036743	total: 5.92s	remaining: 6.84s
    464:	learn: 2.8007923	total: 5.95s	remaining: 6.84s
    465:	learn: 2.7986187	total: 5.96s	remaining: 6.83s
    466:	learn: 2.7975223	total: 5.97s	remaining: 6.82s
    467:	learn: 2.7965000	total: 5.98s	remaining: 6.8s
    468:	learn: 2.7952756	total: 6s	remaining: 6.79s
    469:	learn: 2.7943657	total: 6.01s	remaining: 6.78s
    470:	learn: 2.7937059	total: 6.02s	remaining: 6.76s
    471:	learn: 2.7930171	total: 6.03s	remaining: 6.75s
    472:	learn: 2.7897797	total: 6.04s	remaining: 6.73s
    473:	learn: 2.7884395	total: 6.05s	remaining: 6.72s
    474:	learn: 2.7868242	total: 6.07s	remaining: 6.71s
    475:	learn: 2.7844967	total: 6.08s	remaining: 6.7s
    476:	learn: 2.7839320	total: 6.09s	remaining: 6.68s
    477:	learn: 2.7835713	total: 6.11s	remaining: 6.67s
    478:	learn: 2.7832595	total: 6.12s	remaining: 6.66s
    479:	learn: 2.7828449	total: 6.13s	remaining: 6.64s
    480:	learn: 2.7811264	total: 6.14s	remaining: 6.63s
    481:	learn: 2.7792146	total: 6.15s	remaining: 6.61s
    482:	learn: 2.7775151	total: 6.16s	remaining: 6.6s
    483:	learn: 2.7749513	total: 6.18s	remaining: 6.58s
    484:	learn: 2.7737769	total: 6.19s	remaining: 6.57s
    485:	learn: 2.7709479	total: 6.2s	remaining: 6.56s
    486:	learn: 2.7687288	total: 6.21s	remaining: 6.54s
    487:	learn: 2.7681414	total: 6.22s	remaining: 6.53s
    488:	learn: 2.7677354	total: 6.24s	remaining: 6.52s
    489:	learn: 2.7675566	total: 6.25s	remaining: 6.51s
    490:	learn: 2.7663916	total: 6.26s	remaining: 6.49s
    491:	learn: 2.7650020	total: 6.27s	remaining: 6.48s
    492:	learn: 2.7644360	total: 6.29s	remaining: 6.46s
    493:	learn: 2.7632123	total: 6.3s	remaining: 6.45s
    494:	learn: 2.7619988	total: 6.31s	remaining: 6.44s
    495:	learn: 2.7614048	total: 6.32s	remaining: 6.42s
    496:	learn: 2.7601923	total: 6.33s	remaining: 6.41s
    497:	learn: 2.7588522	total: 6.35s	remaining: 6.4s
    498:	learn: 2.7573293	total: 6.36s	remaining: 6.38s
    499:	learn: 2.7558833	total: 6.37s	remaining: 6.37s
    500:	learn: 2.7540020	total: 6.38s	remaining: 6.36s
    501:	learn: 2.7519097	total: 6.39s	remaining: 6.34s
    502:	learn: 2.7504485	total: 6.41s	remaining: 6.33s
    503:	learn: 2.7487951	total: 6.42s	remaining: 6.32s
    504:	learn: 2.7483400	total: 6.43s	remaining: 6.3s
    505:	learn: 2.7472169	total: 6.44s	remaining: 6.29s
    506:	learn: 2.7470657	total: 6.45s	remaining: 6.27s
    507:	learn: 2.7468184	total: 6.46s	remaining: 6.26s
    508:	learn: 2.7453815	total: 6.47s	remaining: 6.24s
    509:	learn: 2.7440491	total: 6.49s	remaining: 6.23s
    510:	learn: 2.7432219	total: 6.5s	remaining: 6.22s
    511:	learn: 2.7427010	total: 6.51s	remaining: 6.2s
    512:	learn: 2.7423380	total: 6.52s	remaining: 6.19s
    513:	learn: 2.7406359	total: 6.53s	remaining: 6.17s
    514:	learn: 2.7393106	total: 6.54s	remaining: 6.16s
    515:	learn: 2.7383237	total: 6.55s	remaining: 6.15s
    516:	learn: 2.7380809	total: 6.57s	remaining: 6.13s
    517:	learn: 2.7379997	total: 6.58s	remaining: 6.12s
    518:	learn: 2.7373916	total: 6.59s	remaining: 6.11s
    519:	learn: 2.7363929	total: 6.6s	remaining: 6.09s
    520:	learn: 2.7339462	total: 6.61s	remaining: 6.08s
    521:	learn: 2.7334507	total: 6.62s	remaining: 6.06s
    522:	learn: 2.7331638	total: 6.63s	remaining: 6.05s
    523:	learn: 2.7330390	total: 6.64s	remaining: 6.04s
    524:	learn: 2.7316496	total: 6.66s	remaining: 6.02s
    525:	learn: 2.7312789	total: 6.68s	remaining: 6.02s
    526:	learn: 2.7301570	total: 6.69s	remaining: 6s
    527:	learn: 2.7295360	total: 6.7s	remaining: 5.99s
    528:	learn: 2.7288251	total: 6.71s	remaining: 5.97s
    529:	learn: 2.7279860	total: 6.72s	remaining: 5.96s
    530:	learn: 2.7265809	total: 6.74s	remaining: 5.95s
    531:	learn: 2.7254853	total: 6.75s	remaining: 5.94s
    532:	learn: 2.7243639	total: 6.76s	remaining: 5.92s
    533:	learn: 2.7240346	total: 6.77s	remaining: 5.91s
    534:	learn: 2.7234686	total: 6.79s	remaining: 5.9s
    535:	learn: 2.7226348	total: 6.8s	remaining: 5.88s
    536:	learn: 2.7217125	total: 6.81s	remaining: 5.87s
    537:	learn: 2.7199356	total: 6.82s	remaining: 5.86s
    538:	learn: 2.7190511	total: 6.83s	remaining: 5.84s
    539:	learn: 2.7181162	total: 6.85s	remaining: 5.83s
    540:	learn: 2.7179618	total: 6.86s	remaining: 5.82s
    541:	learn: 2.7170508	total: 6.87s	remaining: 5.81s
    542:	learn: 2.7160635	total: 6.88s	remaining: 5.79s
    543:	learn: 2.7143359	total: 6.9s	remaining: 5.78s
    544:	learn: 2.7136836	total: 6.91s	remaining: 5.77s
    545:	learn: 2.7124850	total: 6.93s	remaining: 5.76s
    546:	learn: 2.7116597	total: 6.94s	remaining: 5.75s
    547:	learn: 2.7101062	total: 6.96s	remaining: 5.74s
    548:	learn: 2.7094795	total: 6.97s	remaining: 5.72s
    549:	learn: 2.7088151	total: 6.98s	remaining: 5.71s
    550:	learn: 2.7060958	total: 6.99s	remaining: 5.7s
    551:	learn: 2.7048500	total: 7.01s	remaining: 5.69s
    552:	learn: 2.7045630	total: 7.02s	remaining: 5.67s
    553:	learn: 2.7039058	total: 7.03s	remaining: 5.66s
    554:	learn: 2.7012217	total: 7.04s	remaining: 5.65s
    555:	learn: 2.7004385	total: 7.06s	remaining: 5.63s
    556:	learn: 2.6994008	total: 7.07s	remaining: 5.62s
    557:	learn: 2.6965856	total: 7.09s	remaining: 5.62s
    558:	learn: 2.6961286	total: 7.11s	remaining: 5.61s
    559:	learn: 2.6947527	total: 7.12s	remaining: 5.59s
    560:	learn: 2.6944228	total: 7.13s	remaining: 5.58s
    561:	learn: 2.6923130	total: 7.14s	remaining: 5.57s
    562:	learn: 2.6916425	total: 7.15s	remaining: 5.55s
    563:	learn: 2.6906535	total: 7.16s	remaining: 5.54s
    564:	learn: 2.6889785	total: 7.18s	remaining: 5.53s
    565:	learn: 2.6877226	total: 7.19s	remaining: 5.51s
    566:	learn: 2.6863257	total: 7.2s	remaining: 5.5s
    567:	learn: 2.6858690	total: 7.21s	remaining: 5.49s
    568:	learn: 2.6831131	total: 7.22s	remaining: 5.47s
    569:	learn: 2.6821755	total: 7.24s	remaining: 5.46s
    570:	learn: 2.6803385	total: 7.25s	remaining: 5.45s
    571:	learn: 2.6796104	total: 7.26s	remaining: 5.43s
    572:	learn: 2.6780825	total: 7.27s	remaining: 5.42s
    573:	learn: 2.6770051	total: 7.28s	remaining: 5.4s
    574:	learn: 2.6765800	total: 7.29s	remaining: 5.39s
    575:	learn: 2.6761845	total: 7.3s	remaining: 5.38s
    576:	learn: 2.6752988	total: 7.32s	remaining: 5.36s
    577:	learn: 2.6752354	total: 7.33s	remaining: 5.35s
    578:	learn: 2.6745617	total: 7.34s	remaining: 5.34s
    579:	learn: 2.6716107	total: 7.35s	remaining: 5.32s
    580:	learn: 2.6713648	total: 7.36s	remaining: 5.31s
    581:	learn: 2.6700705	total: 7.38s	remaining: 5.3s
    582:	learn: 2.6693086	total: 7.39s	remaining: 5.28s
    583:	learn: 2.6682615	total: 7.4s	remaining: 5.27s
    584:	learn: 2.6677046	total: 7.41s	remaining: 5.26s
    585:	learn: 2.6660801	total: 7.42s	remaining: 5.24s
    586:	learn: 2.6650469	total: 7.43s	remaining: 5.23s
    587:	learn: 2.6637431	total: 7.45s	remaining: 5.22s
    588:	learn: 2.6622032	total: 7.46s	remaining: 5.2s
    589:	learn: 2.6614983	total: 7.47s	remaining: 5.19s
    590:	learn: 2.6597974	total: 7.48s	remaining: 5.18s
    591:	learn: 2.6592591	total: 7.5s	remaining: 5.17s
    592:	learn: 2.6583242	total: 7.51s	remaining: 5.15s
    593:	learn: 2.6574744	total: 7.52s	remaining: 5.14s
    594:	learn: 2.6565472	total: 7.54s	remaining: 5.13s
    595:	learn: 2.6551886	total: 7.55s	remaining: 5.12s
    596:	learn: 2.6541647	total: 7.56s	remaining: 5.1s
    597:	learn: 2.6540484	total: 7.58s	remaining: 5.09s
    598:	learn: 2.6538877	total: 7.59s	remaining: 5.08s
    599:	learn: 2.6529085	total: 7.6s	remaining: 5.07s
    600:	learn: 2.6516615	total: 7.62s	remaining: 5.05s
    601:	learn: 2.6509627	total: 7.63s	remaining: 5.04s
    602:	learn: 2.6492612	total: 7.64s	remaining: 5.03s
    603:	learn: 2.6475029	total: 7.65s	remaining: 5.02s
    604:	learn: 2.6462042	total: 7.67s	remaining: 5.01s
    605:	learn: 2.6449965	total: 7.68s	remaining: 4.99s
    606:	learn: 2.6442212	total: 7.69s	remaining: 4.98s
    607:	learn: 2.6423669	total: 7.72s	remaining: 4.97s
    608:	learn: 2.6419979	total: 7.73s	remaining: 4.96s
    609:	learn: 2.6408468	total: 7.74s	remaining: 4.95s
    610:	learn: 2.6394284	total: 7.76s	remaining: 4.94s
    611:	learn: 2.6382620	total: 7.77s	remaining: 4.92s
    612:	learn: 2.6375312	total: 7.78s	remaining: 4.91s
    613:	learn: 2.6356316	total: 7.79s	remaining: 4.9s
    614:	learn: 2.6354226	total: 7.81s	remaining: 4.89s
    615:	learn: 2.6338452	total: 7.82s	remaining: 4.87s
    616:	learn: 2.6328907	total: 7.83s	remaining: 4.86s
    617:	learn: 2.6321933	total: 7.84s	remaining: 4.85s
    618:	learn: 2.6315130	total: 7.85s	remaining: 4.83s
    619:	learn: 2.6310944	total: 7.87s	remaining: 4.82s
    620:	learn: 2.6305037	total: 7.88s	remaining: 4.81s
    621:	learn: 2.6286862	total: 7.89s	remaining: 4.8s
    622:	learn: 2.6282091	total: 7.91s	remaining: 4.78s
    623:	learn: 2.6275969	total: 7.92s	remaining: 4.77s
    624:	learn: 2.6273184	total: 7.95s	remaining: 4.77s
    625:	learn: 2.6271253	total: 7.96s	remaining: 4.75s
    626:	learn: 2.6261765	total: 7.98s	remaining: 4.75s
    627:	learn: 2.6253086	total: 7.99s	remaining: 4.73s
    628:	learn: 2.6247750	total: 8s	remaining: 4.72s
    629:	learn: 2.6236436	total: 8.01s	remaining: 4.71s
    630:	learn: 2.6229512	total: 8.03s	remaining: 4.69s
    631:	learn: 2.6216827	total: 8.04s	remaining: 4.68s
    632:	learn: 2.6213123	total: 8.05s	remaining: 4.67s
    633:	learn: 2.6197877	total: 8.06s	remaining: 4.65s
    634:	learn: 2.6188981	total: 8.07s	remaining: 4.64s
    635:	learn: 2.6184704	total: 8.08s	remaining: 4.63s
    636:	learn: 2.6178206	total: 8.09s	remaining: 4.61s
    637:	learn: 2.6164544	total: 8.11s	remaining: 4.6s
    638:	learn: 2.6151634	total: 8.13s	remaining: 4.59s
    639:	learn: 2.6147271	total: 8.14s	remaining: 4.58s
    640:	learn: 2.6143313	total: 8.16s	remaining: 4.57s
    641:	learn: 2.6139730	total: 8.17s	remaining: 4.55s
    642:	learn: 2.6130528	total: 8.18s	remaining: 4.54s
    643:	learn: 2.6126921	total: 8.19s	remaining: 4.53s
    644:	learn: 2.6119583	total: 8.2s	remaining: 4.51s
    645:	learn: 2.6117509	total: 8.22s	remaining: 4.5s
    646:	learn: 2.6095432	total: 8.23s	remaining: 4.49s
    647:	learn: 2.6091412	total: 8.24s	remaining: 4.47s
    648:	learn: 2.6076007	total: 8.25s	remaining: 4.46s
    649:	learn: 2.6064540	total: 8.26s	remaining: 4.45s
    650:	learn: 2.6055559	total: 8.27s	remaining: 4.43s
    651:	learn: 2.6052395	total: 8.28s	remaining: 4.42s
    652:	learn: 2.6048745	total: 8.29s	remaining: 4.41s
    653:	learn: 2.6032826	total: 8.31s	remaining: 4.39s
    654:	learn: 2.6028881	total: 8.32s	remaining: 4.38s
    655:	learn: 2.6020206	total: 8.33s	remaining: 4.37s
    656:	learn: 2.5999085	total: 8.34s	remaining: 4.36s
    657:	learn: 2.5991573	total: 8.36s	remaining: 4.34s
    658:	learn: 2.5991167	total: 8.37s	remaining: 4.33s
    659:	learn: 2.5970370	total: 8.38s	remaining: 4.32s
    660:	learn: 2.5960830	total: 8.4s	remaining: 4.31s
    661:	learn: 2.5922292	total: 8.41s	remaining: 4.29s
    662:	learn: 2.5913997	total: 8.42s	remaining: 4.28s
    663:	learn: 2.5900062	total: 8.43s	remaining: 4.27s
    664:	learn: 2.5889666	total: 8.44s	remaining: 4.25s
    665:	learn: 2.5886747	total: 8.45s	remaining: 4.24s
    666:	learn: 2.5877599	total: 8.46s	remaining: 4.23s
    667:	learn: 2.5871411	total: 8.48s	remaining: 4.21s
    668:	learn: 2.5853490	total: 8.49s	remaining: 4.2s
    669:	learn: 2.5847813	total: 8.5s	remaining: 4.19s
    670:	learn: 2.5839810	total: 8.51s	remaining: 4.17s
    671:	learn: 2.5828903	total: 8.52s	remaining: 4.16s
    672:	learn: 2.5821617	total: 8.53s	remaining: 4.15s
    673:	learn: 2.5812394	total: 8.55s	remaining: 4.13s
    674:	learn: 2.5798483	total: 8.56s	remaining: 4.12s
    675:	learn: 2.5790615	total: 8.57s	remaining: 4.11s
    676:	learn: 2.5782478	total: 8.58s	remaining: 4.09s
    677:	learn: 2.5773538	total: 8.59s	remaining: 4.08s
    678:	learn: 2.5767466	total: 8.6s	remaining: 4.07s
    679:	learn: 2.5758252	total: 8.62s	remaining: 4.05s
    680:	learn: 2.5735224	total: 8.63s	remaining: 4.04s
    681:	learn: 2.5720585	total: 8.65s	remaining: 4.04s
    682:	learn: 2.5716047	total: 8.67s	remaining: 4.02s
    683:	learn: 2.5707242	total: 8.68s	remaining: 4.01s
    684:	learn: 2.5696163	total: 8.69s	remaining: 4s
    685:	learn: 2.5683298	total: 8.72s	remaining: 3.99s
    686:	learn: 2.5674756	total: 8.73s	remaining: 3.98s
    687:	learn: 2.5659628	total: 8.75s	remaining: 3.97s
    688:	learn: 2.5646996	total: 8.76s	remaining: 3.95s
    689:	learn: 2.5641974	total: 8.77s	remaining: 3.94s
    690:	learn: 2.5628486	total: 8.8s	remaining: 3.93s
    691:	learn: 2.5614840	total: 8.81s	remaining: 3.92s
    692:	learn: 2.5603410	total: 8.82s	remaining: 3.91s
    693:	learn: 2.5588220	total: 8.83s	remaining: 3.9s
    694:	learn: 2.5582942	total: 8.85s	remaining: 3.88s
    695:	learn: 2.5575427	total: 8.86s	remaining: 3.87s
    696:	learn: 2.5557600	total: 8.87s	remaining: 3.86s
    697:	learn: 2.5541923	total: 8.88s	remaining: 3.84s
    698:	learn: 2.5526411	total: 8.9s	remaining: 3.83s
    699:	learn: 2.5524048	total: 8.91s	remaining: 3.82s
    700:	learn: 2.5517898	total: 8.92s	remaining: 3.81s
    701:	learn: 2.5511128	total: 8.94s	remaining: 3.79s
    702:	learn: 2.5500447	total: 8.95s	remaining: 3.78s
    703:	learn: 2.5492309	total: 8.96s	remaining: 3.77s
    704:	learn: 2.5488338	total: 8.97s	remaining: 3.75s
    705:	learn: 2.5483424	total: 8.98s	remaining: 3.74s
    706:	learn: 2.5481260	total: 9.01s	remaining: 3.73s
    707:	learn: 2.5470246	total: 9.03s	remaining: 3.72s
    708:	learn: 2.5437359	total: 9.04s	remaining: 3.71s
    709:	learn: 2.5423093	total: 9.05s	remaining: 3.69s
    710:	learn: 2.5420425	total: 9.06s	remaining: 3.68s
    711:	learn: 2.5416197	total: 9.08s	remaining: 3.67s
    712:	learn: 2.5407279	total: 9.09s	remaining: 3.66s
    713:	learn: 2.5400960	total: 9.11s	remaining: 3.65s
    714:	learn: 2.5387321	total: 9.12s	remaining: 3.63s
    715:	learn: 2.5382062	total: 9.13s	remaining: 3.62s
    716:	learn: 2.5370809	total: 9.14s	remaining: 3.61s
    717:	learn: 2.5355111	total: 9.15s	remaining: 3.6s
    718:	learn: 2.5351336	total: 9.17s	remaining: 3.58s
    719:	learn: 2.5333464	total: 9.18s	remaining: 3.57s
    720:	learn: 2.5313589	total: 9.19s	remaining: 3.56s
    721:	learn: 2.5311367	total: 9.2s	remaining: 3.54s
    722:	learn: 2.5298265	total: 9.22s	remaining: 3.53s
    723:	learn: 2.5293536	total: 9.23s	remaining: 3.52s
    724:	learn: 2.5273948	total: 9.24s	remaining: 3.5s
    725:	learn: 2.5267785	total: 9.25s	remaining: 3.49s
    726:	learn: 2.5261922	total: 9.28s	remaining: 3.48s
    727:	learn: 2.5256259	total: 9.29s	remaining: 3.47s
    728:	learn: 2.5253448	total: 9.3s	remaining: 3.46s
    729:	learn: 2.5239628	total: 9.32s	remaining: 3.44s
    730:	learn: 2.5236287	total: 9.33s	remaining: 3.43s
    731:	learn: 2.5226455	total: 9.34s	remaining: 3.42s
    732:	learn: 2.5221587	total: 9.35s	remaining: 3.41s
    733:	learn: 2.5208310	total: 9.37s	remaining: 3.39s
    734:	learn: 2.5186808	total: 9.38s	remaining: 3.38s
    735:	learn: 2.5176450	total: 9.4s	remaining: 3.37s
    736:	learn: 2.5159654	total: 9.41s	remaining: 3.36s
    737:	learn: 2.5146335	total: 9.43s	remaining: 3.35s
    738:	learn: 2.5129761	total: 9.44s	remaining: 3.33s
    739:	learn: 2.5125943	total: 9.45s	remaining: 3.32s
    740:	learn: 2.5112581	total: 9.46s	remaining: 3.31s
    741:	learn: 2.5093657	total: 9.47s	remaining: 3.29s
    742:	learn: 2.5092433	total: 9.48s	remaining: 3.28s
    743:	learn: 2.5083406	total: 9.49s	remaining: 3.27s
    744:	learn: 2.5077301	total: 9.52s	remaining: 3.26s
    745:	learn: 2.5073852	total: 9.54s	remaining: 3.25s
    746:	learn: 2.5067744	total: 9.55s	remaining: 3.23s
    747:	learn: 2.5061229	total: 9.56s	remaining: 3.22s
    748:	learn: 2.5052392	total: 9.57s	remaining: 3.21s
    749:	learn: 2.5039155	total: 9.58s	remaining: 3.19s
    750:	learn: 2.5030237	total: 9.6s	remaining: 3.18s
    751:	learn: 2.5022558	total: 9.61s	remaining: 3.17s
    752:	learn: 2.5015899	total: 9.62s	remaining: 3.16s
    753:	learn: 2.5003815	total: 9.63s	remaining: 3.14s
    754:	learn: 2.4996203	total: 9.65s	remaining: 3.13s
    755:	learn: 2.4974987	total: 9.66s	remaining: 3.12s
    756:	learn: 2.4973126	total: 9.67s	remaining: 3.1s
    757:	learn: 2.4959917	total: 9.69s	remaining: 3.09s
    758:	learn: 2.4948153	total: 9.7s	remaining: 3.08s
    759:	learn: 2.4942527	total: 9.72s	remaining: 3.07s
    760:	learn: 2.4935047	total: 9.74s	remaining: 3.06s
    761:	learn: 2.4928818	total: 9.75s	remaining: 3.04s
    762:	learn: 2.4918299	total: 9.77s	remaining: 3.03s
    763:	learn: 2.4908240	total: 9.79s	remaining: 3.02s
    764:	learn: 2.4902078	total: 9.8s	remaining: 3.01s
    765:	learn: 2.4897749	total: 9.81s	remaining: 3s
    766:	learn: 2.4874798	total: 9.82s	remaining: 2.98s
    767:	learn: 2.4871232	total: 9.83s	remaining: 2.97s
    768:	learn: 2.4855785	total: 9.85s	remaining: 2.96s
    769:	learn: 2.4854302	total: 9.86s	remaining: 2.94s
    770:	learn: 2.4849527	total: 9.87s	remaining: 2.93s
    771:	learn: 2.4830288	total: 9.88s	remaining: 2.92s
    772:	learn: 2.4825822	total: 9.89s	remaining: 2.9s
    773:	learn: 2.4823024	total: 9.9s	remaining: 2.89s
    774:	learn: 2.4809409	total: 9.91s	remaining: 2.88s
    775:	learn: 2.4798357	total: 9.93s	remaining: 2.87s
    776:	learn: 2.4787543	total: 9.94s	remaining: 2.85s
    777:	learn: 2.4782731	total: 9.95s	remaining: 2.84s
    778:	learn: 2.4770533	total: 9.96s	remaining: 2.83s
    779:	learn: 2.4769702	total: 9.97s	remaining: 2.81s
    780:	learn: 2.4751248	total: 9.99s	remaining: 2.8s
    781:	learn: 2.4738596	total: 10s	remaining: 2.79s
    782:	learn: 2.4733867	total: 10s	remaining: 2.77s
    783:	learn: 2.4730533	total: 10s	remaining: 2.76s
    784:	learn: 2.4708581	total: 10s	remaining: 2.75s
    785:	learn: 2.4692406	total: 10s	remaining: 2.73s
    786:	learn: 2.4689806	total: 10.1s	remaining: 2.72s
    787:	learn: 2.4686630	total: 10.1s	remaining: 2.71s
    788:	learn: 2.4672965	total: 10.1s	remaining: 2.69s
    789:	learn: 2.4667715	total: 10.1s	remaining: 2.68s
    790:	learn: 2.4655791	total: 10.1s	remaining: 2.67s
    791:	learn: 2.4650812	total: 10.1s	remaining: 2.65s
    792:	learn: 2.4630230	total: 10.1s	remaining: 2.64s
    793:	learn: 2.4621139	total: 10.1s	remaining: 2.63s
    794:	learn: 2.4614099	total: 10.1s	remaining: 2.62s
    795:	learn: 2.4596436	total: 10.2s	remaining: 2.6s
    796:	learn: 2.4578873	total: 10.2s	remaining: 2.59s
    797:	learn: 2.4565844	total: 10.2s	remaining: 2.58s
    798:	learn: 2.4562897	total: 10.2s	remaining: 2.57s
    799:	learn: 2.4559001	total: 10.2s	remaining: 2.55s
    800:	learn: 2.4546551	total: 10.2s	remaining: 2.54s
    801:	learn: 2.4529291	total: 10.2s	remaining: 2.53s
    802:	learn: 2.4522086	total: 10.2s	remaining: 2.51s
    803:	learn: 2.4519196	total: 10.3s	remaining: 2.5s
    804:	learn: 2.4518211	total: 10.3s	remaining: 2.49s
    805:	learn: 2.4494265	total: 10.3s	remaining: 2.48s
    806:	learn: 2.4490557	total: 10.3s	remaining: 2.46s
    807:	learn: 2.4478612	total: 10.3s	remaining: 2.45s
    808:	learn: 2.4463830	total: 10.3s	remaining: 2.44s
    809:	learn: 2.4452156	total: 10.3s	remaining: 2.42s
    810:	learn: 2.4450027	total: 10.4s	remaining: 2.41s
    811:	learn: 2.4441426	total: 10.4s	remaining: 2.4s
    812:	learn: 2.4425621	total: 10.4s	remaining: 2.39s
    813:	learn: 2.4421185	total: 10.4s	remaining: 2.37s
    814:	learn: 2.4404535	total: 10.4s	remaining: 2.36s
    815:	learn: 2.4398047	total: 10.4s	remaining: 2.35s
    816:	learn: 2.4382389	total: 10.4s	remaining: 2.33s
    817:	learn: 2.4380353	total: 10.4s	remaining: 2.32s
    818:	learn: 2.4379604	total: 10.5s	remaining: 2.31s
    819:	learn: 2.4370209	total: 10.5s	remaining: 2.3s
    820:	learn: 2.4368578	total: 10.5s	remaining: 2.29s
    821:	learn: 2.4364433	total: 10.5s	remaining: 2.27s
    822:	learn: 2.4361422	total: 10.5s	remaining: 2.26s
    823:	learn: 2.4354292	total: 10.5s	remaining: 2.25s
    824:	learn: 2.4337606	total: 10.5s	remaining: 2.23s
    825:	learn: 2.4329191	total: 10.5s	remaining: 2.22s
    826:	learn: 2.4324971	total: 10.5s	remaining: 2.21s
    827:	learn: 2.4321565	total: 10.6s	remaining: 2.19s
    828:	learn: 2.4310011	total: 10.6s	remaining: 2.18s
    829:	learn: 2.4301478	total: 10.6s	remaining: 2.17s
    830:	learn: 2.4293959	total: 10.6s	remaining: 2.15s
    831:	learn: 2.4275226	total: 10.6s	remaining: 2.14s
    832:	learn: 2.4264596	total: 10.6s	remaining: 2.13s
    833:	learn: 2.4261876	total: 10.6s	remaining: 2.12s
    834:	learn: 2.4247538	total: 10.6s	remaining: 2.1s
    835:	learn: 2.4241255	total: 10.7s	remaining: 2.09s
    836:	learn: 2.4239324	total: 10.7s	remaining: 2.08s
    837:	learn: 2.4230032	total: 10.7s	remaining: 2.06s
    838:	learn: 2.4214302	total: 10.7s	remaining: 2.05s
    839:	learn: 2.4202766	total: 10.7s	remaining: 2.04s
    840:	learn: 2.4200276	total: 10.7s	remaining: 2.02s
    841:	learn: 2.4192948	total: 10.7s	remaining: 2.01s
    842:	learn: 2.4189759	total: 10.7s	remaining: 2s
    843:	learn: 2.4182000	total: 10.7s	remaining: 1.98s
    844:	learn: 2.4180276	total: 10.8s	remaining: 1.97s
    845:	learn: 2.4170992	total: 10.8s	remaining: 1.96s
    846:	learn: 2.4167308	total: 10.8s	remaining: 1.95s
    847:	learn: 2.4154037	total: 10.8s	remaining: 1.93s
    848:	learn: 2.4138015	total: 10.8s	remaining: 1.92s
    849:	learn: 2.4129681	total: 10.8s	remaining: 1.91s
    850:	learn: 2.4121085	total: 10.8s	remaining: 1.89s
    851:	learn: 2.4115139	total: 10.8s	remaining: 1.88s
    852:	learn: 2.4109882	total: 10.8s	remaining: 1.87s
    853:	learn: 2.4106154	total: 10.9s	remaining: 1.85s
    854:	learn: 2.4104304	total: 10.9s	remaining: 1.84s
    855:	learn: 2.4102906	total: 10.9s	remaining: 1.83s
    856:	learn: 2.4100124	total: 10.9s	remaining: 1.82s
    857:	learn: 2.4093098	total: 10.9s	remaining: 1.8s
    858:	learn: 2.4089428	total: 10.9s	remaining: 1.79s
    859:	learn: 2.4065732	total: 11s	remaining: 1.78s
    860:	learn: 2.4062307	total: 11s	remaining: 1.77s
    861:	learn: 2.4056008	total: 11s	remaining: 1.76s
    862:	learn: 2.4050882	total: 11s	remaining: 1.75s
    863:	learn: 2.4048459	total: 11s	remaining: 1.73s
    864:	learn: 2.4045165	total: 11s	remaining: 1.72s
    865:	learn: 2.4041634	total: 11s	remaining: 1.71s
    866:	learn: 2.4030699	total: 11s	remaining: 1.69s
    867:	learn: 2.4016540	total: 11.1s	remaining: 1.68s
    868:	learn: 2.4005930	total: 11.1s	remaining: 1.67s
    869:	learn: 2.4004315	total: 11.1s	remaining: 1.66s
    870:	learn: 2.3996678	total: 11.1s	remaining: 1.64s
    871:	learn: 2.3984852	total: 11.1s	remaining: 1.63s
    872:	learn: 2.3964986	total: 11.1s	remaining: 1.62s
    873:	learn: 2.3956061	total: 11.1s	remaining: 1.6s
    874:	learn: 2.3947244	total: 11.1s	remaining: 1.59s
    875:	learn: 2.3939817	total: 11.1s	remaining: 1.58s
    876:	learn: 2.3938981	total: 11.2s	remaining: 1.56s
    877:	learn: 2.3922752	total: 11.2s	remaining: 1.55s
    878:	learn: 2.3910319	total: 11.2s	remaining: 1.54s
    879:	learn: 2.3907826	total: 11.2s	remaining: 1.53s
    880:	learn: 2.3892318	total: 11.2s	remaining: 1.51s
    881:	learn: 2.3886935	total: 11.2s	remaining: 1.5s
    882:	learn: 2.3876057	total: 11.2s	remaining: 1.49s
    883:	learn: 2.3869282	total: 11.2s	remaining: 1.48s
    884:	learn: 2.3864128	total: 11.3s	remaining: 1.46s
    885:	learn: 2.3862135	total: 11.3s	remaining: 1.45s
    886:	learn: 2.3841142	total: 11.3s	remaining: 1.44s
    887:	learn: 2.3836825	total: 11.3s	remaining: 1.42s
    888:	learn: 2.3828384	total: 11.3s	remaining: 1.41s
    889:	learn: 2.3821463	total: 11.3s	remaining: 1.4s
    890:	learn: 2.3811726	total: 11.3s	remaining: 1.39s
    891:	learn: 2.3792819	total: 11.3s	remaining: 1.37s
    892:	learn: 2.3781479	total: 11.4s	remaining: 1.36s
    893:	learn: 2.3776955	total: 11.4s	remaining: 1.35s
    894:	learn: 2.3771746	total: 11.4s	remaining: 1.33s
    895:	learn: 2.3764317	total: 11.4s	remaining: 1.32s
    896:	learn: 2.3756007	total: 11.4s	remaining: 1.31s
    897:	learn: 2.3747101	total: 11.4s	remaining: 1.3s
    898:	learn: 2.3745607	total: 11.4s	remaining: 1.28s
    899:	learn: 2.3739202	total: 11.4s	remaining: 1.27s
    900:	learn: 2.3734929	total: 11.4s	remaining: 1.26s
    901:	learn: 2.3731379	total: 11.5s	remaining: 1.25s
    902:	learn: 2.3727611	total: 11.5s	remaining: 1.23s
    903:	learn: 2.3725090	total: 11.5s	remaining: 1.22s
    904:	learn: 2.3717098	total: 11.5s	remaining: 1.21s
    905:	learn: 2.3706591	total: 11.5s	remaining: 1.19s
    906:	learn: 2.3703851	total: 11.5s	remaining: 1.18s
    907:	learn: 2.3682291	total: 11.5s	remaining: 1.17s
    908:	learn: 2.3679546	total: 11.6s	remaining: 1.16s
    909:	learn: 2.3667257	total: 11.6s	remaining: 1.14s
    910:	learn: 2.3661046	total: 11.6s	remaining: 1.13s
    911:	learn: 2.3659651	total: 11.6s	remaining: 1.12s
    912:	learn: 2.3648620	total: 11.6s	remaining: 1.1s
    913:	learn: 2.3629870	total: 11.6s	remaining: 1.09s
    914:	learn: 2.3628074	total: 11.6s	remaining: 1.08s
    915:	learn: 2.3623246	total: 11.7s	remaining: 1.07s
    916:	learn: 2.3619470	total: 11.7s	remaining: 1.06s
    917:	learn: 2.3614876	total: 11.7s	remaining: 1.04s
    918:	learn: 2.3597153	total: 11.7s	remaining: 1.03s
    919:	learn: 2.3589373	total: 11.7s	remaining: 1.02s
    920:	learn: 2.3586718	total: 11.7s	remaining: 1s
    921:	learn: 2.3583987	total: 11.7s	remaining: 993ms
    922:	learn: 2.3583100	total: 11.7s	remaining: 980ms
    923:	learn: 2.3581948	total: 11.8s	remaining: 967ms
    924:	learn: 2.3576880	total: 11.8s	remaining: 954ms
    925:	learn: 2.3573336	total: 11.8s	remaining: 942ms
    926:	learn: 2.3571494	total: 11.8s	remaining: 929ms
    927:	learn: 2.3563707	total: 11.8s	remaining: 916ms
    928:	learn: 2.3554626	total: 11.8s	remaining: 903ms
    929:	learn: 2.3551484	total: 11.8s	remaining: 890ms
    930:	learn: 2.3544299	total: 11.8s	remaining: 878ms
    931:	learn: 2.3529985	total: 11.8s	remaining: 865ms
    932:	learn: 2.3514477	total: 11.9s	remaining: 852ms
    933:	learn: 2.3510502	total: 11.9s	remaining: 839ms
    934:	learn: 2.3500177	total: 11.9s	remaining: 826ms
    935:	learn: 2.3481350	total: 11.9s	remaining: 813ms
    936:	learn: 2.3472794	total: 11.9s	remaining: 801ms
    937:	learn: 2.3471623	total: 11.9s	remaining: 788ms
    938:	learn: 2.3452765	total: 11.9s	remaining: 775ms
    939:	learn: 2.3451540	total: 11.9s	remaining: 762ms
    940:	learn: 2.3442740	total: 12s	remaining: 750ms
    941:	learn: 2.3441803	total: 12s	remaining: 737ms
    942:	learn: 2.3434243	total: 12s	remaining: 725ms
    943:	learn: 2.3432749	total: 12s	remaining: 712ms
    944:	learn: 2.3426499	total: 12s	remaining: 699ms
    945:	learn: 2.3416084	total: 12s	remaining: 686ms
    946:	learn: 2.3413577	total: 12s	remaining: 674ms
    947:	learn: 2.3403413	total: 12s	remaining: 661ms
    948:	learn: 2.3399036	total: 12.1s	remaining: 648ms
    949:	learn: 2.3395801	total: 12.1s	remaining: 635ms
    950:	learn: 2.3386936	total: 12.1s	remaining: 622ms
    951:	learn: 2.3376372	total: 12.1s	remaining: 610ms
    952:	learn: 2.3367671	total: 12.1s	remaining: 597ms
    953:	learn: 2.3359051	total: 12.1s	remaining: 584ms
    954:	learn: 2.3353176	total: 12.1s	remaining: 571ms
    955:	learn: 2.3347031	total: 12.1s	remaining: 559ms
    956:	learn: 2.3335848	total: 12.1s	remaining: 546ms
    957:	learn: 2.3325824	total: 12.2s	remaining: 533ms
    958:	learn: 2.3324498	total: 12.2s	remaining: 520ms
    959:	learn: 2.3315113	total: 12.2s	remaining: 508ms
    960:	learn: 2.3301714	total: 12.2s	remaining: 495ms
    961:	learn: 2.3296696	total: 12.2s	remaining: 482ms
    962:	learn: 2.3293956	total: 12.2s	remaining: 469ms
    963:	learn: 2.3291049	total: 12.2s	remaining: 457ms
    964:	learn: 2.3272404	total: 12.2s	remaining: 444ms
    965:	learn: 2.3256960	total: 12.3s	remaining: 431ms
    966:	learn: 2.3251016	total: 12.3s	remaining: 419ms
    967:	learn: 2.3244167	total: 12.3s	remaining: 406ms
    968:	learn: 2.3241374	total: 12.3s	remaining: 394ms
    969:	learn: 2.3223916	total: 12.3s	remaining: 381ms
    970:	learn: 2.3220974	total: 12.3s	remaining: 368ms
    971:	learn: 2.3217939	total: 12.3s	remaining: 355ms
    972:	learn: 2.3206335	total: 12.3s	remaining: 343ms
    973:	learn: 2.3203285	total: 12.4s	remaining: 330ms
    974:	learn: 2.3193375	total: 12.4s	remaining: 317ms
    975:	learn: 2.3189236	total: 12.4s	remaining: 305ms
    976:	learn: 2.3181204	total: 12.4s	remaining: 292ms
    977:	learn: 2.3165660	total: 12.4s	remaining: 279ms
    978:	learn: 2.3160011	total: 12.4s	remaining: 266ms
    979:	learn: 2.3155241	total: 12.4s	remaining: 254ms
    980:	learn: 2.3141821	total: 12.4s	remaining: 241ms
    981:	learn: 2.3138816	total: 12.5s	remaining: 228ms
    982:	learn: 2.3131255	total: 12.5s	remaining: 216ms
    983:	learn: 2.3126353	total: 12.5s	remaining: 203ms
    984:	learn: 2.3125351	total: 12.5s	remaining: 190ms
    985:	learn: 2.3123311	total: 12.5s	remaining: 178ms
    986:	learn: 2.3114021	total: 12.5s	remaining: 165ms
    987:	learn: 2.3110838	total: 12.5s	remaining: 152ms
    988:	learn: 2.3109927	total: 12.5s	remaining: 139ms
    989:	learn: 2.3099468	total: 12.5s	remaining: 127ms
    990:	learn: 2.3089665	total: 12.6s	remaining: 114ms
    991:	learn: 2.3083197	total: 12.6s	remaining: 101ms
    992:	learn: 2.3076627	total: 12.6s	remaining: 88.7ms
    993:	learn: 2.3069271	total: 12.6s	remaining: 76.1ms
    994:	learn: 2.3061408	total: 12.6s	remaining: 63.4ms
    995:	learn: 2.3059026	total: 12.6s	remaining: 50.7ms
    996:	learn: 2.3052528	total: 12.6s	remaining: 38ms
    997:	learn: 2.3043241	total: 12.6s	remaining: 25.3ms
    998:	learn: 2.3035092	total: 12.7s	remaining: 12.7ms
    999:	learn: 2.3029046	total: 12.7s	remaining: 0us
    




    <catboost.core.CatBoostRegressor at 0x14dd826b550>




```python
y_pred_val = model.predict(X_val)
```


```python
r2_score(y_pred_val,y_val)
```




    0.8263408163975153




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

    Learning rate set to 0.045547
    0:	learn: 6.5894424	total: 12.1ms	remaining: 12.1s
    1:	learn: 6.4097736	total: 23.9ms	remaining: 11.9s
    2:	learn: 6.2407225	total: 35.3ms	remaining: 11.7s
    3:	learn: 6.0876786	total: 62.1ms	remaining: 15.5s
    4:	learn: 5.9353770	total: 72.7ms	remaining: 14.5s
    5:	learn: 5.7948122	total: 82.9ms	remaining: 13.7s
    6:	learn: 5.6640716	total: 94.4ms	remaining: 13.4s
    7:	learn: 5.5369636	total: 105ms	remaining: 13s
    8:	learn: 5.4172900	total: 123ms	remaining: 13.5s
    9:	learn: 5.3021201	total: 131ms	remaining: 13s
    10:	learn: 5.1912358	total: 141ms	remaining: 12.7s
    11:	learn: 5.0936174	total: 154ms	remaining: 12.7s
    12:	learn: 5.0064458	total: 165ms	remaining: 12.6s
    13:	learn: 4.9195349	total: 174ms	remaining: 12.2s
    14:	learn: 4.8285895	total: 199ms	remaining: 13.1s
    15:	learn: 4.7514596	total: 212ms	remaining: 13s
    16:	learn: 4.6707595	total: 223ms	remaining: 12.9s
    17:	learn: 4.6058973	total: 244ms	remaining: 13.3s
    18:	learn: 4.5477375	total: 250ms	remaining: 12.9s
    19:	learn: 4.4787885	total: 260ms	remaining: 12.8s
    20:	learn: 4.4275662	total: 268ms	remaining: 12.5s
    21:	learn: 4.3926859	total: 272ms	remaining: 12.1s
    22:	learn: 4.3606567	total: 276ms	remaining: 11.7s
    23:	learn: 4.3042638	total: 286ms	remaining: 11.6s
    24:	learn: 4.2671225	total: 291ms	remaining: 11.3s
    25:	learn: 4.2223249	total: 300ms	remaining: 11.2s
    26:	learn: 4.1876631	total: 311ms	remaining: 11.2s
    27:	learn: 4.1498471	total: 338ms	remaining: 11.7s
    28:	learn: 4.1215277	total: 351ms	remaining: 11.8s
    29:	learn: 4.0822743	total: 369ms	remaining: 11.9s
    30:	learn: 4.0515858	total: 379ms	remaining: 11.9s
    31:	learn: 4.0229315	total: 399ms	remaining: 12.1s
    32:	learn: 3.9998965	total: 406ms	remaining: 11.9s
    33:	learn: 3.9684043	total: 418ms	remaining: 11.9s
    34:	learn: 3.9425507	total: 428ms	remaining: 11.8s
    35:	learn: 3.9133832	total: 447ms	remaining: 12s
    36:	learn: 3.9022726	total: 451ms	remaining: 11.7s
    37:	learn: 3.8727406	total: 466ms	remaining: 11.8s
    38:	learn: 3.8514490	total: 476ms	remaining: 11.7s
    39:	learn: 3.8258984	total: 487ms	remaining: 11.7s
    40:	learn: 3.8080156	total: 508ms	remaining: 11.9s
    41:	learn: 3.8006986	total: 512ms	remaining: 11.7s
    42:	learn: 3.7847405	total: 522ms	remaining: 11.6s
    43:	learn: 3.7717174	total: 532ms	remaining: 11.6s
    44:	learn: 3.7540891	total: 555ms	remaining: 11.8s
    45:	learn: 3.7415606	total: 570ms	remaining: 11.8s
    46:	learn: 3.7251482	total: 581ms	remaining: 11.8s
    47:	learn: 3.7122982	total: 600ms	remaining: 11.9s
    48:	learn: 3.7075800	total: 613ms	remaining: 11.9s
    49:	learn: 3.7036214	total: 626ms	remaining: 11.9s
    50:	learn: 3.6920444	total: 637ms	remaining: 11.9s
    51:	learn: 3.6868495	total: 658ms	remaining: 12s
    52:	learn: 3.6770191	total: 677ms	remaining: 12.1s
    53:	learn: 3.6639391	total: 687ms	remaining: 12s
    54:	learn: 3.6528796	total: 726ms	remaining: 12.5s
    55:	learn: 3.6397690	total: 738ms	remaining: 12.4s
    56:	learn: 3.6265931	total: 758ms	remaining: 12.5s
    57:	learn: 3.6219104	total: 765ms	remaining: 12.4s
    58:	learn: 3.6129141	total: 788ms	remaining: 12.6s
    59:	learn: 3.6012095	total: 803ms	remaining: 12.6s
    60:	learn: 3.5950202	total: 815ms	remaining: 12.6s
    61:	learn: 3.5881703	total: 830ms	remaining: 12.6s
    62:	learn: 3.5864944	total: 834ms	remaining: 12.4s
    63:	learn: 3.5779694	total: 852ms	remaining: 12.5s
    64:	learn: 3.5693745	total: 862ms	remaining: 12.4s
    65:	learn: 3.5638092	total: 872ms	remaining: 12.3s
    66:	learn: 3.5550514	total: 882ms	remaining: 12.3s
    67:	learn: 3.5539600	total: 886ms	remaining: 12.1s
    68:	learn: 3.5477770	total: 898ms	remaining: 12.1s
    69:	learn: 3.5434427	total: 908ms	remaining: 12.1s
    70:	learn: 3.5374451	total: 919ms	remaining: 12s
    71:	learn: 3.5315971	total: 930ms	remaining: 12s
    72:	learn: 3.5255082	total: 940ms	remaining: 11.9s
    73:	learn: 3.5199025	total: 951ms	remaining: 11.9s
    74:	learn: 3.5130610	total: 961ms	remaining: 11.9s
    75:	learn: 3.5115331	total: 972ms	remaining: 11.8s
    76:	learn: 3.5080165	total: 988ms	remaining: 11.8s
    77:	learn: 3.5049308	total: 999ms	remaining: 11.8s
    78:	learn: 3.4985082	total: 1.02s	remaining: 11.9s
    79:	learn: 3.4937707	total: 1.03s	remaining: 11.9s
    80:	learn: 3.4921361	total: 1.05s	remaining: 11.9s
    81:	learn: 3.4860183	total: 1.06s	remaining: 11.9s
    82:	learn: 3.4802427	total: 1.07s	remaining: 11.9s
    83:	learn: 3.4759916	total: 1.09s	remaining: 11.8s
    84:	learn: 3.4745909	total: 1.1s	remaining: 11.8s
    85:	learn: 3.4715401	total: 1.11s	remaining: 11.8s
    86:	learn: 3.4671252	total: 1.13s	remaining: 11.9s
    87:	learn: 3.4664052	total: 1.14s	remaining: 11.8s
    88:	learn: 3.4620939	total: 1.15s	remaining: 11.8s
    89:	learn: 3.4600262	total: 1.18s	remaining: 11.9s
    90:	learn: 3.4581920	total: 1.19s	remaining: 11.9s
    91:	learn: 3.4539907	total: 1.21s	remaining: 11.9s
    92:	learn: 3.4513749	total: 1.22s	remaining: 11.9s
    93:	learn: 3.4494141	total: 1.24s	remaining: 11.9s
    94:	learn: 3.4477028	total: 1.25s	remaining: 12s
    95:	learn: 3.4462280	total: 1.26s	remaining: 11.9s
    96:	learn: 3.4443905	total: 1.29s	remaining: 12s
    97:	learn: 3.4413581	total: 1.3s	remaining: 12s
    98:	learn: 3.4401696	total: 1.31s	remaining: 11.9s
    99:	learn: 3.4396514	total: 1.32s	remaining: 11.9s
    100:	learn: 3.4365160	total: 1.33s	remaining: 11.9s
    101:	learn: 3.4331298	total: 1.35s	remaining: 11.9s
    102:	learn: 3.4289514	total: 1.36s	remaining: 11.8s
    103:	learn: 3.4263676	total: 1.37s	remaining: 11.8s
    104:	learn: 3.4212283	total: 1.4s	remaining: 11.9s
    105:	learn: 3.4178453	total: 1.41s	remaining: 11.9s
    106:	learn: 3.4163717	total: 1.43s	remaining: 11.9s
    107:	learn: 3.4123764	total: 1.44s	remaining: 11.9s
    108:	learn: 3.4112792	total: 1.45s	remaining: 11.8s
    109:	learn: 3.4046481	total: 1.46s	remaining: 11.8s
    110:	learn: 3.4015750	total: 1.47s	remaining: 11.8s
    111:	learn: 3.3978997	total: 1.49s	remaining: 11.8s
    112:	learn: 3.3931485	total: 1.5s	remaining: 11.8s
    113:	learn: 3.3929330	total: 1.51s	remaining: 11.7s
    114:	learn: 3.3923176	total: 1.51s	remaining: 11.7s
    115:	learn: 3.3918933	total: 1.52s	remaining: 11.6s
    116:	learn: 3.3908908	total: 1.53s	remaining: 11.6s
    117:	learn: 3.3891986	total: 1.54s	remaining: 11.5s
    118:	learn: 3.3876885	total: 1.57s	remaining: 11.6s
    119:	learn: 3.3860822	total: 1.58s	remaining: 11.6s
    120:	learn: 3.3840390	total: 1.59s	remaining: 11.6s
    121:	learn: 3.3832520	total: 1.6s	remaining: 11.5s
    122:	learn: 3.3813733	total: 1.61s	remaining: 11.5s
    123:	learn: 3.3789812	total: 1.63s	remaining: 11.5s
    124:	learn: 3.3780588	total: 1.64s	remaining: 11.5s
    125:	learn: 3.3740397	total: 1.66s	remaining: 11.5s
    126:	learn: 3.3720866	total: 1.68s	remaining: 11.5s
    127:	learn: 3.3713207	total: 1.69s	remaining: 11.5s
    128:	learn: 3.3690875	total: 1.71s	remaining: 11.6s
    129:	learn: 3.3673001	total: 1.72s	remaining: 11.5s
    130:	learn: 3.3648389	total: 1.73s	remaining: 11.5s
    131:	learn: 3.3629723	total: 1.74s	remaining: 11.5s
    132:	learn: 3.3607757	total: 1.75s	remaining: 11.4s
    133:	learn: 3.3589443	total: 1.77s	remaining: 11.4s
    134:	learn: 3.3565446	total: 1.78s	remaining: 11.4s
    135:	learn: 3.3543584	total: 1.79s	remaining: 11.4s
    136:	learn: 3.3499946	total: 1.8s	remaining: 11.4s
    137:	learn: 3.3488486	total: 1.82s	remaining: 11.3s
    138:	learn: 3.3456836	total: 1.83s	remaining: 11.3s
    139:	learn: 3.3424225	total: 1.84s	remaining: 11.3s
    140:	learn: 3.3412244	total: 1.85s	remaining: 11.2s
    141:	learn: 3.3385074	total: 1.86s	remaining: 11.2s
    142:	learn: 3.3363756	total: 1.87s	remaining: 11.2s
    143:	learn: 3.3362572	total: 1.87s	remaining: 11.1s
    144:	learn: 3.3344909	total: 1.88s	remaining: 11.1s
    145:	learn: 3.3326466	total: 1.9s	remaining: 11.1s
    146:	learn: 3.3307482	total: 1.91s	remaining: 11.1s
    147:	learn: 3.3304677	total: 1.91s	remaining: 11s
    148:	learn: 3.3259274	total: 1.93s	remaining: 11s
    149:	learn: 3.3239563	total: 1.94s	remaining: 11s
    150:	learn: 3.3214787	total: 1.95s	remaining: 11s
    151:	learn: 3.3182177	total: 1.96s	remaining: 10.9s
    152:	learn: 3.3169422	total: 1.97s	remaining: 10.9s
    153:	learn: 3.3167808	total: 1.98s	remaining: 10.9s
    154:	learn: 3.3144707	total: 2s	remaining: 10.9s
    155:	learn: 3.3134789	total: 2.01s	remaining: 10.9s
    156:	learn: 3.3110855	total: 2.02s	remaining: 10.9s
    157:	learn: 3.3109162	total: 2.03s	remaining: 10.8s
    158:	learn: 3.3097359	total: 2.04s	remaining: 10.8s
    159:	learn: 3.3097288	total: 2.04s	remaining: 10.7s
    160:	learn: 3.3090963	total: 2.05s	remaining: 10.7s
    161:	learn: 3.3070422	total: 2.07s	remaining: 10.7s
    162:	learn: 3.3059762	total: 2.08s	remaining: 10.7s
    163:	learn: 3.3053515	total: 2.09s	remaining: 10.7s
    164:	learn: 3.3044940	total: 2.1s	remaining: 10.6s
    165:	learn: 3.3026183	total: 2.11s	remaining: 10.6s
    166:	learn: 3.3013111	total: 2.13s	remaining: 10.6s
    167:	learn: 3.3002674	total: 2.14s	remaining: 10.6s
    168:	learn: 3.2993586	total: 2.15s	remaining: 10.6s
    169:	learn: 3.2955814	total: 2.16s	remaining: 10.6s
    170:	learn: 3.2945457	total: 2.17s	remaining: 10.5s
    171:	learn: 3.2942770	total: 2.18s	remaining: 10.5s
    172:	learn: 3.2927223	total: 2.19s	remaining: 10.5s
    173:	learn: 3.2887356	total: 2.21s	remaining: 10.5s
    174:	learn: 3.2876685	total: 2.22s	remaining: 10.5s
    175:	learn: 3.2861446	total: 2.24s	remaining: 10.5s
    176:	learn: 3.2824777	total: 2.25s	remaining: 10.5s
    177:	learn: 3.2824758	total: 2.25s	remaining: 10.4s
    178:	learn: 3.2811132	total: 2.27s	remaining: 10.4s
    179:	learn: 3.2795315	total: 2.28s	remaining: 10.4s
    180:	learn: 3.2794694	total: 2.29s	remaining: 10.4s
    181:	learn: 3.2776126	total: 2.31s	remaining: 10.4s
    182:	learn: 3.2769697	total: 2.32s	remaining: 10.3s
    183:	learn: 3.2749829	total: 2.33s	remaining: 10.3s
    184:	learn: 3.2733830	total: 2.34s	remaining: 10.3s
    185:	learn: 3.2728801	total: 2.36s	remaining: 10.3s
    186:	learn: 3.2706524	total: 2.37s	remaining: 10.3s
    187:	learn: 3.2697276	total: 2.38s	remaining: 10.3s
    188:	learn: 3.2697270	total: 2.39s	remaining: 10.2s
    189:	learn: 3.2664738	total: 2.4s	remaining: 10.2s
    190:	learn: 3.2661292	total: 2.41s	remaining: 10.2s
    191:	learn: 3.2655507	total: 2.42s	remaining: 10.2s
    192:	learn: 3.2634022	total: 2.43s	remaining: 10.2s
    193:	learn: 3.2605085	total: 2.44s	remaining: 10.2s
    194:	learn: 3.2585657	total: 2.46s	remaining: 10.1s
    195:	learn: 3.2578208	total: 2.47s	remaining: 10.1s
    196:	learn: 3.2574003	total: 2.48s	remaining: 10.1s
    197:	learn: 3.2558639	total: 2.49s	remaining: 10.1s
    198:	learn: 3.2540547	total: 2.51s	remaining: 10.1s
    199:	learn: 3.2527714	total: 2.52s	remaining: 10.1s
    200:	learn: 3.2518585	total: 2.56s	remaining: 10.2s
    201:	learn: 3.2513845	total: 2.57s	remaining: 10.2s
    202:	learn: 3.2507166	total: 2.58s	remaining: 10.1s
    203:	learn: 3.2494441	total: 2.61s	remaining: 10.2s
    204:	learn: 3.2490625	total: 2.62s	remaining: 10.2s
    205:	learn: 3.2468818	total: 2.63s	remaining: 10.1s
    206:	learn: 3.2443411	total: 2.64s	remaining: 10.1s
    207:	learn: 3.2439201	total: 2.67s	remaining: 10.2s
    208:	learn: 3.2395748	total: 2.68s	remaining: 10.1s
    209:	learn: 3.2385063	total: 2.7s	remaining: 10.2s
    210:	learn: 3.2379185	total: 2.71s	remaining: 10.1s
    211:	learn: 3.2373434	total: 2.72s	remaining: 10.1s
    212:	learn: 3.2357988	total: 2.73s	remaining: 10.1s
    213:	learn: 3.2347445	total: 2.74s	remaining: 10.1s
    214:	learn: 3.2347442	total: 2.75s	remaining: 10s
    215:	learn: 3.2319266	total: 2.76s	remaining: 10s
    216:	learn: 3.2291757	total: 2.78s	remaining: 10s
    217:	learn: 3.2282244	total: 2.79s	remaining: 10s
    218:	learn: 3.2262483	total: 2.81s	remaining: 10s
    219:	learn: 3.2242454	total: 2.83s	remaining: 10s
    220:	learn: 3.2227253	total: 2.84s	remaining: 10s
    221:	learn: 3.2195701	total: 2.86s	remaining: 10s
    222:	learn: 3.2180021	total: 2.87s	remaining: 9.99s
    223:	learn: 3.2158562	total: 2.88s	remaining: 9.97s
    224:	learn: 3.2151899	total: 2.89s	remaining: 9.95s
    225:	learn: 3.2132059	total: 2.9s	remaining: 9.95s
    226:	learn: 3.2126151	total: 2.91s	remaining: 9.93s
    227:	learn: 3.2112519	total: 2.93s	remaining: 9.91s
    228:	learn: 3.2102517	total: 2.94s	remaining: 9.89s
    229:	learn: 3.2055414	total: 2.95s	remaining: 9.88s
    230:	learn: 3.2036843	total: 2.96s	remaining: 9.87s
    231:	learn: 3.2015656	total: 2.98s	remaining: 9.87s
    232:	learn: 3.2003709	total: 2.99s	remaining: 9.85s
    233:	learn: 3.1983376	total: 3.01s	remaining: 9.86s
    234:	learn: 3.1964588	total: 3.02s	remaining: 9.84s
    235:	learn: 3.1959034	total: 3.03s	remaining: 9.82s
    236:	learn: 3.1948378	total: 3.04s	remaining: 9.8s
    237:	learn: 3.1936817	total: 3.05s	remaining: 9.78s
    238:	learn: 3.1929718	total: 3.06s	remaining: 9.76s
    239:	learn: 3.1918896	total: 3.07s	remaining: 9.73s
    240:	learn: 3.1875355	total: 3.08s	remaining: 9.72s
    241:	learn: 3.1871645	total: 3.11s	remaining: 9.73s
    242:	learn: 3.1848972	total: 3.12s	remaining: 9.72s
    243:	learn: 3.1804948	total: 3.13s	remaining: 9.71s
    244:	learn: 3.1797925	total: 3.14s	remaining: 9.69s
    245:	learn: 3.1789430	total: 3.16s	remaining: 9.67s
    246:	learn: 3.1751867	total: 3.17s	remaining: 9.66s
    247:	learn: 3.1738046	total: 3.18s	remaining: 9.65s
    248:	learn: 3.1726732	total: 3.19s	remaining: 9.63s
    249:	learn: 3.1706614	total: 3.21s	remaining: 9.62s
    250:	learn: 3.1699471	total: 3.22s	remaining: 9.6s
    251:	learn: 3.1678401	total: 3.23s	remaining: 9.58s
    252:	learn: 3.1652718	total: 3.24s	remaining: 9.56s
    253:	learn: 3.1631699	total: 3.26s	remaining: 9.58s
    254:	learn: 3.1620358	total: 3.28s	remaining: 9.58s
    255:	learn: 3.1604623	total: 3.29s	remaining: 9.58s
    256:	learn: 3.1587201	total: 3.32s	remaining: 9.6s
    257:	learn: 3.1572518	total: 3.35s	remaining: 9.64s
    258:	learn: 3.1557504	total: 3.36s	remaining: 9.62s
    259:	learn: 3.1546006	total: 3.4s	remaining: 9.68s
    260:	learn: 3.1534743	total: 3.42s	remaining: 9.67s
    261:	learn: 3.1526466	total: 3.42s	remaining: 9.65s
    262:	learn: 3.1516092	total: 3.44s	remaining: 9.63s
    263:	learn: 3.1502948	total: 3.45s	remaining: 9.61s
    264:	learn: 3.1480620	total: 3.46s	remaining: 9.59s
    265:	learn: 3.1445847	total: 3.47s	remaining: 9.57s
    266:	learn: 3.1443310	total: 3.49s	remaining: 9.59s
    267:	learn: 3.1431576	total: 3.5s	remaining: 9.57s
    268:	learn: 3.1418659	total: 3.52s	remaining: 9.55s
    269:	learn: 3.1403589	total: 3.53s	remaining: 9.54s
    270:	learn: 3.1380697	total: 3.54s	remaining: 9.52s
    271:	learn: 3.1372027	total: 3.55s	remaining: 9.51s
    272:	learn: 3.1366674	total: 3.56s	remaining: 9.49s
    273:	learn: 3.1356204	total: 3.57s	remaining: 9.47s
    274:	learn: 3.1336981	total: 3.59s	remaining: 9.46s
    275:	learn: 3.1331313	total: 3.61s	remaining: 9.48s
    276:	learn: 3.1328176	total: 3.63s	remaining: 9.48s
    277:	learn: 3.1323679	total: 3.65s	remaining: 9.47s
    278:	learn: 3.1315713	total: 3.66s	remaining: 9.45s
    279:	learn: 3.1306668	total: 3.68s	remaining: 9.46s
    280:	learn: 3.1295490	total: 3.71s	remaining: 9.48s
    281:	learn: 3.1288165	total: 3.72s	remaining: 9.46s
    282:	learn: 3.1277688	total: 3.73s	remaining: 9.45s
    283:	learn: 3.1267184	total: 3.75s	remaining: 9.46s
    284:	learn: 3.1248197	total: 3.77s	remaining: 9.45s
    285:	learn: 3.1217162	total: 3.78s	remaining: 9.44s
    286:	learn: 3.1189905	total: 3.79s	remaining: 9.42s
    287:	learn: 3.1181769	total: 3.81s	remaining: 9.43s
    288:	learn: 3.1151009	total: 3.85s	remaining: 9.46s
    289:	learn: 3.1137128	total: 3.86s	remaining: 9.44s
    290:	learn: 3.1113675	total: 3.88s	remaining: 9.45s
    291:	learn: 3.1094492	total: 3.89s	remaining: 9.43s
    292:	learn: 3.1084358	total: 3.9s	remaining: 9.41s
    293:	learn: 3.1069012	total: 3.92s	remaining: 9.42s
    294:	learn: 3.1058812	total: 3.94s	remaining: 9.41s
    295:	learn: 3.1048415	total: 3.95s	remaining: 9.4s
    296:	learn: 3.1044995	total: 3.98s	remaining: 9.43s
    297:	learn: 3.1033086	total: 3.99s	remaining: 9.41s
    298:	learn: 3.1024816	total: 4s	remaining: 9.39s
    299:	learn: 3.1002686	total: 4.03s	remaining: 9.4s
    300:	learn: 3.0965375	total: 4.04s	remaining: 9.39s
    301:	learn: 3.0946998	total: 4.06s	remaining: 9.38s
    302:	learn: 3.0941182	total: 4.07s	remaining: 9.36s
    303:	learn: 3.0920908	total: 4.09s	remaining: 9.37s
    304:	learn: 3.0919388	total: 4.1s	remaining: 9.35s
    305:	learn: 3.0892315	total: 4.13s	remaining: 9.36s
    306:	learn: 3.0887789	total: 4.13s	remaining: 9.34s
    307:	learn: 3.0871657	total: 4.15s	remaining: 9.32s
    308:	learn: 3.0863523	total: 4.16s	remaining: 9.3s
    309:	learn: 3.0841307	total: 4.17s	remaining: 9.28s
    310:	learn: 3.0829615	total: 4.18s	remaining: 9.27s
    311:	learn: 3.0813444	total: 4.19s	remaining: 9.25s
    312:	learn: 3.0799883	total: 4.21s	remaining: 9.23s
    313:	learn: 3.0787786	total: 4.22s	remaining: 9.21s
    314:	learn: 3.0774240	total: 4.23s	remaining: 9.21s
    315:	learn: 3.0756912	total: 4.25s	remaining: 9.2s
    316:	learn: 3.0752817	total: 4.26s	remaining: 9.18s
    317:	learn: 3.0748356	total: 4.28s	remaining: 9.17s
    318:	learn: 3.0741360	total: 4.29s	remaining: 9.17s
    319:	learn: 3.0728687	total: 4.3s	remaining: 9.15s
    320:	learn: 3.0717030	total: 4.33s	remaining: 9.15s
    321:	learn: 3.0706185	total: 4.34s	remaining: 9.13s
    322:	learn: 3.0686309	total: 4.36s	remaining: 9.13s
    323:	learn: 3.0679975	total: 4.37s	remaining: 9.12s
    324:	learn: 3.0678177	total: 4.39s	remaining: 9.11s
    325:	learn: 3.0674284	total: 4.4s	remaining: 9.09s
    326:	learn: 3.0662501	total: 4.42s	remaining: 9.1s
    327:	learn: 3.0652387	total: 4.43s	remaining: 9.08s
    328:	learn: 3.0646688	total: 4.45s	remaining: 9.07s
    329:	learn: 3.0633163	total: 4.46s	remaining: 9.06s
    330:	learn: 3.0604422	total: 4.48s	remaining: 9.06s
    331:	learn: 3.0594046	total: 4.5s	remaining: 9.04s
    332:	learn: 3.0584118	total: 4.51s	remaining: 9.04s
    333:	learn: 3.0580066	total: 4.53s	remaining: 9.02s
    334:	learn: 3.0567219	total: 4.56s	remaining: 9.05s
    335:	learn: 3.0546638	total: 4.58s	remaining: 9.04s
    336:	learn: 3.0539321	total: 4.59s	remaining: 9.03s
    337:	learn: 3.0516227	total: 4.6s	remaining: 9.01s
    338:	learn: 3.0513034	total: 4.61s	remaining: 8.99s
    339:	learn: 3.0487835	total: 4.62s	remaining: 8.97s
    340:	learn: 3.0479753	total: 4.63s	remaining: 8.95s
    341:	learn: 3.0474775	total: 4.64s	remaining: 8.93s
    342:	learn: 3.0463576	total: 4.65s	remaining: 8.91s
    343:	learn: 3.0460637	total: 4.67s	remaining: 8.9s
    344:	learn: 3.0455853	total: 4.68s	remaining: 8.89s
    345:	learn: 3.0426669	total: 4.69s	remaining: 8.87s
    346:	learn: 3.0403612	total: 4.7s	remaining: 8.85s
    347:	learn: 3.0397168	total: 4.71s	remaining: 8.83s
    348:	learn: 3.0382808	total: 4.73s	remaining: 8.82s
    349:	learn: 3.0372394	total: 4.74s	remaining: 8.8s
    350:	learn: 3.0351040	total: 4.75s	remaining: 8.79s
    351:	learn: 3.0342025	total: 4.76s	remaining: 8.77s
    352:	learn: 3.0335761	total: 4.79s	remaining: 8.78s
    353:	learn: 3.0328760	total: 4.8s	remaining: 8.77s
    354:	learn: 3.0314557	total: 4.82s	remaining: 8.75s
    355:	learn: 3.0302934	total: 4.83s	remaining: 8.73s
    356:	learn: 3.0276576	total: 4.84s	remaining: 8.71s
    357:	learn: 3.0274563	total: 4.85s	remaining: 8.7s
    358:	learn: 3.0259810	total: 4.86s	remaining: 8.68s
    359:	learn: 3.0228122	total: 4.87s	remaining: 8.66s
    360:	learn: 3.0226071	total: 4.88s	remaining: 8.64s
    361:	learn: 3.0212843	total: 4.9s	remaining: 8.63s
    362:	learn: 3.0204538	total: 4.91s	remaining: 8.62s
    363:	learn: 3.0200191	total: 4.93s	remaining: 8.61s
    364:	learn: 3.0195021	total: 4.94s	remaining: 8.59s
    365:	learn: 3.0191176	total: 4.95s	remaining: 8.57s
    366:	learn: 3.0181236	total: 4.96s	remaining: 8.56s
    367:	learn: 3.0175638	total: 4.97s	remaining: 8.54s
    368:	learn: 3.0163999	total: 4.99s	remaining: 8.54s
    369:	learn: 3.0145593	total: 5s	remaining: 8.52s
    370:	learn: 3.0138104	total: 5.01s	remaining: 8.49s
    371:	learn: 3.0109806	total: 5.04s	remaining: 8.5s
    372:	learn: 3.0105956	total: 5.05s	remaining: 8.48s
    373:	learn: 3.0091153	total: 5.06s	remaining: 8.47s
    374:	learn: 3.0072984	total: 5.08s	remaining: 8.47s
    375:	learn: 3.0058834	total: 5.09s	remaining: 8.46s
    376:	learn: 3.0052710	total: 5.14s	remaining: 8.5s
    377:	learn: 3.0048095	total: 5.16s	remaining: 8.49s
    378:	learn: 3.0031999	total: 5.17s	remaining: 8.47s
    379:	learn: 3.0018908	total: 5.19s	remaining: 8.47s
    380:	learn: 3.0004723	total: 5.2s	remaining: 8.46s
    381:	learn: 2.9980500	total: 5.21s	remaining: 8.44s
    382:	learn: 2.9967520	total: 5.24s	remaining: 8.44s
    383:	learn: 2.9959725	total: 5.25s	remaining: 8.42s
    384:	learn: 2.9939468	total: 5.26s	remaining: 8.4s
    385:	learn: 2.9916186	total: 5.27s	remaining: 8.39s
    386:	learn: 2.9907155	total: 5.3s	remaining: 8.4s
    387:	learn: 2.9874695	total: 5.31s	remaining: 8.38s
    388:	learn: 2.9858715	total: 5.32s	remaining: 8.36s
    389:	learn: 2.9840844	total: 5.33s	remaining: 8.34s
    390:	learn: 2.9831583	total: 5.34s	remaining: 8.32s
    391:	learn: 2.9823822	total: 5.36s	remaining: 8.31s
    392:	learn: 2.9818706	total: 5.37s	remaining: 8.29s
    393:	learn: 2.9813569	total: 5.39s	remaining: 8.29s
    394:	learn: 2.9794755	total: 5.41s	remaining: 8.28s
    395:	learn: 2.9753497	total: 5.42s	remaining: 8.27s
    396:	learn: 2.9739822	total: 5.43s	remaining: 8.25s
    397:	learn: 2.9719198	total: 5.45s	remaining: 8.25s
    398:	learn: 2.9712177	total: 5.46s	remaining: 8.23s
    399:	learn: 2.9707906	total: 5.48s	remaining: 8.23s
    400:	learn: 2.9689256	total: 5.5s	remaining: 8.21s
    401:	learn: 2.9668918	total: 5.51s	remaining: 8.2s
    402:	learn: 2.9662031	total: 5.52s	remaining: 8.18s
    403:	learn: 2.9639712	total: 5.54s	remaining: 8.17s
    404:	learn: 2.9627205	total: 5.55s	remaining: 8.15s
    405:	learn: 2.9624870	total: 5.56s	remaining: 8.13s
    406:	learn: 2.9593874	total: 5.57s	remaining: 8.12s
    407:	learn: 2.9575866	total: 5.58s	remaining: 8.1s
    408:	learn: 2.9559585	total: 5.6s	remaining: 8.09s
    409:	learn: 2.9554811	total: 5.61s	remaining: 8.07s
    410:	learn: 2.9551585	total: 5.62s	remaining: 8.05s
    411:	learn: 2.9530220	total: 5.63s	remaining: 8.04s
    412:	learn: 2.9514012	total: 5.65s	remaining: 8.04s
    413:	learn: 2.9479545	total: 5.68s	remaining: 8.04s
    414:	learn: 2.9475006	total: 5.7s	remaining: 8.03s
    415:	learn: 2.9461646	total: 5.71s	remaining: 8.02s
    416:	learn: 2.9437208	total: 5.72s	remaining: 8s
    417:	learn: 2.9411854	total: 5.73s	remaining: 7.98s
    418:	learn: 2.9401755	total: 5.74s	remaining: 7.96s
    419:	learn: 2.9393832	total: 5.75s	remaining: 7.95s
    420:	learn: 2.9378703	total: 5.79s	remaining: 7.96s
    421:	learn: 2.9366638	total: 5.8s	remaining: 7.95s
    422:	learn: 2.9338685	total: 5.81s	remaining: 7.93s
    423:	learn: 2.9331876	total: 5.82s	remaining: 7.91s
    424:	learn: 2.9316718	total: 5.84s	remaining: 7.9s
    425:	learn: 2.9301612	total: 5.85s	remaining: 7.88s
    426:	learn: 2.9298773	total: 5.87s	remaining: 7.88s
    427:	learn: 2.9285511	total: 5.88s	remaining: 7.86s
    428:	learn: 2.9277015	total: 5.92s	remaining: 7.87s
    429:	learn: 2.9256695	total: 5.92s	remaining: 7.85s
    430:	learn: 2.9244348	total: 5.93s	remaining: 7.84s
    431:	learn: 2.9235061	total: 5.95s	remaining: 7.82s
    432:	learn: 2.9228860	total: 5.96s	remaining: 7.8s
    433:	learn: 2.9221795	total: 5.97s	remaining: 7.78s
    434:	learn: 2.9204582	total: 5.98s	remaining: 7.77s
    435:	learn: 2.9188464	total: 5.99s	remaining: 7.75s
    436:	learn: 2.9181115	total: 6s	remaining: 7.73s
    437:	learn: 2.9163059	total: 6.03s	remaining: 7.73s
    438:	learn: 2.9143928	total: 6.04s	remaining: 7.71s
    439:	learn: 2.9116807	total: 6.05s	remaining: 7.71s
    440:	learn: 2.9102736	total: 6.07s	remaining: 7.69s
    441:	learn: 2.9082283	total: 6.08s	remaining: 7.68s
    442:	learn: 2.9069851	total: 6.1s	remaining: 7.67s
    443:	learn: 2.9062197	total: 6.11s	remaining: 7.65s
    444:	learn: 2.9042888	total: 6.13s	remaining: 7.65s
    445:	learn: 2.9032951	total: 6.14s	remaining: 7.63s
    446:	learn: 2.9021495	total: 6.15s	remaining: 7.61s
    447:	learn: 2.9007426	total: 6.16s	remaining: 7.6s
    448:	learn: 2.8987278	total: 6.18s	remaining: 7.58s
    449:	learn: 2.8977398	total: 6.19s	remaining: 7.56s
    450:	learn: 2.8968695	total: 6.2s	remaining: 7.54s
    451:	learn: 2.8956879	total: 6.21s	remaining: 7.53s
    452:	learn: 2.8944372	total: 6.22s	remaining: 7.51s
    453:	learn: 2.8938675	total: 6.23s	remaining: 7.49s
    454:	learn: 2.8933635	total: 6.24s	remaining: 7.47s
    455:	learn: 2.8928259	total: 6.25s	remaining: 7.45s
    456:	learn: 2.8915410	total: 6.27s	remaining: 7.45s
    457:	learn: 2.8909278	total: 6.29s	remaining: 7.44s
    458:	learn: 2.8893743	total: 6.3s	remaining: 7.43s
    459:	learn: 2.8884088	total: 6.31s	remaining: 7.41s
    460:	learn: 2.8879965	total: 6.32s	remaining: 7.39s
    461:	learn: 2.8865997	total: 6.35s	remaining: 7.39s
    462:	learn: 2.8857018	total: 6.36s	remaining: 7.38s
    463:	learn: 2.8845841	total: 6.38s	remaining: 7.36s
    464:	learn: 2.8840830	total: 6.39s	remaining: 7.35s
    465:	learn: 2.8815041	total: 6.41s	remaining: 7.34s
    466:	learn: 2.8811124	total: 6.42s	remaining: 7.33s
    467:	learn: 2.8799028	total: 6.43s	remaining: 7.31s
    468:	learn: 2.8791853	total: 6.45s	remaining: 7.31s
    469:	learn: 2.8783170	total: 6.47s	remaining: 7.3s
    470:	learn: 2.8770448	total: 6.48s	remaining: 7.28s
    471:	learn: 2.8745541	total: 6.49s	remaining: 7.26s
    472:	learn: 2.8741073	total: 6.5s	remaining: 7.25s
    473:	learn: 2.8737446	total: 6.52s	remaining: 7.23s
    474:	learn: 2.8730694	total: 6.53s	remaining: 7.22s
    475:	learn: 2.8712851	total: 6.55s	remaining: 7.21s
    476:	learn: 2.8702641	total: 6.56s	remaining: 7.2s
    477:	learn: 2.8694347	total: 6.57s	remaining: 7.18s
    478:	learn: 2.8685806	total: 6.58s	remaining: 7.16s
    479:	learn: 2.8679774	total: 6.59s	remaining: 7.14s
    480:	learn: 2.8669836	total: 6.61s	remaining: 7.13s
    481:	learn: 2.8655117	total: 6.62s	remaining: 7.11s
    482:	learn: 2.8642996	total: 6.64s	remaining: 7.11s
    483:	learn: 2.8628859	total: 6.65s	remaining: 7.09s
    484:	learn: 2.8606949	total: 6.66s	remaining: 7.08s
    485:	learn: 2.8597181	total: 6.67s	remaining: 7.06s
    486:	learn: 2.8587512	total: 6.68s	remaining: 7.04s
    487:	learn: 2.8577756	total: 6.7s	remaining: 7.03s
    488:	learn: 2.8566592	total: 6.71s	remaining: 7.01s
    489:	learn: 2.8564980	total: 6.72s	remaining: 7s
    490:	learn: 2.8559620	total: 6.73s	remaining: 6.98s
    491:	learn: 2.8558191	total: 6.75s	remaining: 6.97s
    492:	learn: 2.8553065	total: 6.76s	remaining: 6.95s
    493:	learn: 2.8539472	total: 6.77s	remaining: 6.93s
    494:	learn: 2.8528893	total: 6.78s	remaining: 6.92s
    495:	learn: 2.8517684	total: 6.79s	remaining: 6.9s
    496:	learn: 2.8507092	total: 6.81s	remaining: 6.89s
    497:	learn: 2.8487195	total: 6.83s	remaining: 6.88s
    498:	learn: 2.8481333	total: 6.83s	remaining: 6.86s
    499:	learn: 2.8474281	total: 6.84s	remaining: 6.84s
    500:	learn: 2.8472583	total: 6.86s	remaining: 6.83s
    501:	learn: 2.8468823	total: 6.87s	remaining: 6.81s
    502:	learn: 2.8449961	total: 6.88s	remaining: 6.79s
    503:	learn: 2.8442308	total: 6.88s	remaining: 6.78s
    504:	learn: 2.8430850	total: 6.89s	remaining: 6.76s
    505:	learn: 2.8411007	total: 6.92s	remaining: 6.75s
    506:	learn: 2.8395474	total: 6.93s	remaining: 6.74s
    507:	learn: 2.8386719	total: 6.94s	remaining: 6.72s
    508:	learn: 2.8359700	total: 6.95s	remaining: 6.71s
    509:	learn: 2.8345054	total: 6.96s	remaining: 6.69s
    510:	learn: 2.8336927	total: 6.97s	remaining: 6.67s
    511:	learn: 2.8321723	total: 6.98s	remaining: 6.65s
    512:	learn: 2.8305809	total: 6.99s	remaining: 6.64s
    513:	learn: 2.8287985	total: 7.01s	remaining: 6.63s
    514:	learn: 2.8265679	total: 7.02s	remaining: 6.62s
    515:	learn: 2.8261242	total: 7.04s	remaining: 6.6s
    516:	learn: 2.8243404	total: 7.05s	remaining: 6.59s
    517:	learn: 2.8231952	total: 7.07s	remaining: 6.58s
    518:	learn: 2.8223792	total: 7.09s	remaining: 6.57s
    519:	learn: 2.8212820	total: 7.1s	remaining: 6.55s
    520:	learn: 2.8188056	total: 7.11s	remaining: 6.53s
    521:	learn: 2.8160824	total: 7.12s	remaining: 6.52s
    522:	learn: 2.8142526	total: 7.13s	remaining: 6.5s
    523:	learn: 2.8127111	total: 7.15s	remaining: 6.49s
    524:	learn: 2.8107538	total: 7.16s	remaining: 6.48s
    525:	learn: 2.8102938	total: 7.18s	remaining: 6.47s
    526:	learn: 2.8084609	total: 7.2s	remaining: 6.46s
    527:	learn: 2.8073863	total: 7.21s	remaining: 6.44s
    528:	learn: 2.8055819	total: 7.22s	remaining: 6.43s
    529:	learn: 2.8040237	total: 7.23s	remaining: 6.41s
    530:	learn: 2.8028279	total: 7.24s	remaining: 6.4s
    531:	learn: 2.8018277	total: 7.25s	remaining: 6.38s
    532:	learn: 2.8009897	total: 7.26s	remaining: 6.36s
    533:	learn: 2.8004242	total: 7.28s	remaining: 6.35s
    534:	learn: 2.8003431	total: 7.3s	remaining: 6.34s
    535:	learn: 2.7982278	total: 7.32s	remaining: 6.33s
    536:	learn: 2.7977821	total: 7.33s	remaining: 6.32s
    537:	learn: 2.7962785	total: 7.35s	remaining: 6.31s
    538:	learn: 2.7947086	total: 7.36s	remaining: 6.3s
    539:	learn: 2.7941389	total: 7.38s	remaining: 6.29s
    540:	learn: 2.7931306	total: 7.39s	remaining: 6.27s
    541:	learn: 2.7924038	total: 7.41s	remaining: 6.26s
    542:	learn: 2.7903217	total: 7.43s	remaining: 6.25s
    543:	learn: 2.7884312	total: 7.44s	remaining: 6.24s
    544:	learn: 2.7870915	total: 7.45s	remaining: 6.22s
    545:	learn: 2.7864621	total: 7.47s	remaining: 6.21s
    546:	learn: 2.7858498	total: 7.48s	remaining: 6.2s
    547:	learn: 2.7853238	total: 7.49s	remaining: 6.18s
    548:	learn: 2.7838691	total: 7.5s	remaining: 6.17s
    549:	learn: 2.7829868	total: 7.51s	remaining: 6.15s
    550:	learn: 2.7816385	total: 7.53s	remaining: 6.14s
    551:	learn: 2.7813899	total: 7.54s	remaining: 6.12s
    552:	learn: 2.7812831	total: 7.55s	remaining: 6.11s
    553:	learn: 2.7804214	total: 7.57s	remaining: 6.09s
    554:	learn: 2.7798974	total: 7.58s	remaining: 6.08s
    555:	learn: 2.7780852	total: 7.62s	remaining: 6.08s
    556:	learn: 2.7777457	total: 7.63s	remaining: 6.07s
    557:	learn: 2.7774398	total: 7.64s	remaining: 6.05s
    558:	learn: 2.7768387	total: 7.66s	remaining: 6.04s
    559:	learn: 2.7758617	total: 7.67s	remaining: 6.03s
    560:	learn: 2.7751755	total: 7.69s	remaining: 6.01s
    561:	learn: 2.7744904	total: 7.7s	remaining: 6s
    562:	learn: 2.7739510	total: 7.71s	remaining: 5.98s
    563:	learn: 2.7737192	total: 7.72s	remaining: 5.97s
    564:	learn: 2.7710599	total: 7.73s	remaining: 5.95s
    565:	learn: 2.7691292	total: 7.75s	remaining: 5.94s
    566:	learn: 2.7685801	total: 7.76s	remaining: 5.93s
    567:	learn: 2.7684957	total: 7.78s	remaining: 5.91s
    568:	learn: 2.7662201	total: 7.79s	remaining: 5.9s
    569:	learn: 2.7656065	total: 7.81s	remaining: 5.89s
    570:	learn: 2.7638456	total: 7.82s	remaining: 5.88s
    571:	learn: 2.7620121	total: 7.83s	remaining: 5.86s
    572:	learn: 2.7609801	total: 7.84s	remaining: 5.84s
    573:	learn: 2.7596677	total: 7.85s	remaining: 5.83s
    574:	learn: 2.7583547	total: 7.87s	remaining: 5.82s
    575:	learn: 2.7577228	total: 7.88s	remaining: 5.8s
    576:	learn: 2.7565620	total: 7.89s	remaining: 5.79s
    577:	learn: 2.7551787	total: 7.9s	remaining: 5.77s
    578:	learn: 2.7546365	total: 7.91s	remaining: 5.75s
    579:	learn: 2.7542605	total: 7.92s	remaining: 5.74s
    580:	learn: 2.7541684	total: 7.93s	remaining: 5.72s
    581:	learn: 2.7536642	total: 7.94s	remaining: 5.71s
    582:	learn: 2.7534111	total: 7.96s	remaining: 5.7s
    583:	learn: 2.7521427	total: 7.97s	remaining: 5.68s
    584:	learn: 2.7517113	total: 7.98s	remaining: 5.66s
    585:	learn: 2.7501235	total: 8s	remaining: 5.65s
    586:	learn: 2.7499905	total: 8s	remaining: 5.63s
    587:	learn: 2.7477835	total: 8.02s	remaining: 5.62s
    588:	learn: 2.7465231	total: 8.04s	remaining: 5.61s
    589:	learn: 2.7458517	total: 8.05s	remaining: 5.59s
    590:	learn: 2.7456149	total: 8.07s	remaining: 5.58s
    591:	learn: 2.7449792	total: 8.08s	remaining: 5.57s
    592:	learn: 2.7442592	total: 8.09s	remaining: 5.55s
    593:	learn: 2.7436039	total: 8.1s	remaining: 5.54s
    594:	learn: 2.7431375	total: 8.11s	remaining: 5.52s
    595:	learn: 2.7415228	total: 8.12s	remaining: 5.5s
    596:	learn: 2.7403993	total: 8.15s	remaining: 5.5s
    597:	learn: 2.7390201	total: 8.16s	remaining: 5.48s
    598:	learn: 2.7380946	total: 8.18s	remaining: 5.47s
    599:	learn: 2.7361930	total: 8.19s	remaining: 5.46s
    600:	learn: 2.7352812	total: 8.2s	remaining: 5.44s
    601:	learn: 2.7336796	total: 8.21s	remaining: 5.43s
    602:	learn: 2.7331371	total: 8.23s	remaining: 5.42s
    603:	learn: 2.7318495	total: 8.25s	remaining: 5.41s
    604:	learn: 2.7304557	total: 8.27s	remaining: 5.4s
    605:	learn: 2.7287447	total: 8.29s	remaining: 5.39s
    606:	learn: 2.7280081	total: 8.3s	remaining: 5.37s
    607:	learn: 2.7274903	total: 8.31s	remaining: 5.36s
    608:	learn: 2.7265586	total: 8.32s	remaining: 5.34s
    609:	learn: 2.7241940	total: 8.33s	remaining: 5.33s
    610:	learn: 2.7235779	total: 8.35s	remaining: 5.32s
    611:	learn: 2.7229284	total: 8.36s	remaining: 5.3s
    612:	learn: 2.7217949	total: 8.38s	remaining: 5.29s
    613:	learn: 2.7206932	total: 8.39s	remaining: 5.28s
    614:	learn: 2.7200331	total: 8.4s	remaining: 5.26s
    615:	learn: 2.7193435	total: 8.43s	remaining: 5.25s
    616:	learn: 2.7187281	total: 8.44s	remaining: 5.24s
    617:	learn: 2.7181263	total: 8.45s	remaining: 5.22s
    618:	learn: 2.7175978	total: 8.47s	remaining: 5.21s
    619:	learn: 2.7169995	total: 8.49s	remaining: 5.2s
    620:	learn: 2.7168535	total: 8.5s	remaining: 5.19s
    621:	learn: 2.7158567	total: 8.52s	remaining: 5.18s
    622:	learn: 2.7151118	total: 8.53s	remaining: 5.16s
    623:	learn: 2.7147150	total: 8.54s	remaining: 5.15s
    624:	learn: 2.7132307	total: 8.55s	remaining: 5.13s
    625:	learn: 2.7117103	total: 8.56s	remaining: 5.12s
    626:	learn: 2.7106208	total: 8.57s	remaining: 5.1s
    627:	learn: 2.7096946	total: 8.59s	remaining: 5.08s
    628:	learn: 2.7094074	total: 8.6s	remaining: 5.07s
    629:	learn: 2.7089384	total: 8.61s	remaining: 5.05s
    630:	learn: 2.7074399	total: 8.62s	remaining: 5.04s
    631:	learn: 2.7062564	total: 8.63s	remaining: 5.02s
    632:	learn: 2.7045002	total: 8.64s	remaining: 5.01s
    633:	learn: 2.7035472	total: 8.65s	remaining: 5s
    634:	learn: 2.7029397	total: 8.66s	remaining: 4.98s
    635:	learn: 2.7010039	total: 8.68s	remaining: 4.96s
    636:	learn: 2.6997528	total: 8.69s	remaining: 4.95s
    637:	learn: 2.6983598	total: 8.7s	remaining: 4.94s
    638:	learn: 2.6973903	total: 8.72s	remaining: 4.92s
    639:	learn: 2.6950054	total: 8.73s	remaining: 4.91s
    640:	learn: 2.6943744	total: 8.75s	remaining: 4.9s
    641:	learn: 2.6930865	total: 8.76s	remaining: 4.88s
    642:	learn: 2.6926158	total: 8.77s	remaining: 4.87s
    643:	learn: 2.6922486	total: 8.79s	remaining: 4.86s
    644:	learn: 2.6917651	total: 8.8s	remaining: 4.84s
    645:	learn: 2.6911631	total: 8.81s	remaining: 4.83s
    646:	learn: 2.6901925	total: 8.82s	remaining: 4.81s
    647:	learn: 2.6889089	total: 8.84s	remaining: 4.8s
    648:	learn: 2.6885853	total: 8.85s	remaining: 4.79s
    649:	learn: 2.6878191	total: 8.87s	remaining: 4.77s
    650:	learn: 2.6865028	total: 8.89s	remaining: 4.76s
    651:	learn: 2.6860822	total: 8.9s	remaining: 4.75s
    652:	learn: 2.6853869	total: 8.91s	remaining: 4.73s
    653:	learn: 2.6851078	total: 8.92s	remaining: 4.72s
    654:	learn: 2.6847317	total: 8.93s	remaining: 4.7s
    655:	learn: 2.6842258	total: 8.95s	remaining: 4.69s
    656:	learn: 2.6828646	total: 8.96s	remaining: 4.68s
    657:	learn: 2.6815762	total: 8.98s	remaining: 4.67s
    658:	learn: 2.6810284	total: 9s	remaining: 4.66s
    659:	learn: 2.6807132	total: 9.01s	remaining: 4.64s
    660:	learn: 2.6798025	total: 9.03s	remaining: 4.63s
    661:	learn: 2.6788709	total: 9.04s	remaining: 4.62s
    662:	learn: 2.6780282	total: 9.05s	remaining: 4.6s
    663:	learn: 2.6771626	total: 9.07s	remaining: 4.59s
    664:	learn: 2.6754931	total: 9.09s	remaining: 4.58s
    665:	learn: 2.6752521	total: 9.1s	remaining: 4.56s
    666:	learn: 2.6746797	total: 9.11s	remaining: 4.55s
    667:	learn: 2.6735948	total: 9.12s	remaining: 4.53s
    668:	learn: 2.6729864	total: 9.13s	remaining: 4.52s
    669:	learn: 2.6721100	total: 9.15s	remaining: 4.51s
    670:	learn: 2.6707612	total: 9.16s	remaining: 4.49s
    671:	learn: 2.6701978	total: 9.17s	remaining: 4.48s
    672:	learn: 2.6693341	total: 9.18s	remaining: 4.46s
    673:	learn: 2.6691060	total: 9.19s	remaining: 4.45s
    674:	learn: 2.6674127	total: 9.21s	remaining: 4.43s
    675:	learn: 2.6667041	total: 9.22s	remaining: 4.42s
    676:	learn: 2.6656916	total: 9.24s	remaining: 4.41s
    677:	learn: 2.6652088	total: 9.26s	remaining: 4.4s
    678:	learn: 2.6645976	total: 9.27s	remaining: 4.38s
    679:	learn: 2.6639724	total: 9.28s	remaining: 4.37s
    680:	learn: 2.6635223	total: 9.29s	remaining: 4.35s
    681:	learn: 2.6624257	total: 9.31s	remaining: 4.34s
    682:	learn: 2.6620566	total: 9.32s	remaining: 4.32s
    683:	learn: 2.6617626	total: 9.33s	remaining: 4.31s
    684:	learn: 2.6602411	total: 9.34s	remaining: 4.29s
    685:	learn: 2.6579682	total: 9.35s	remaining: 4.28s
    686:	learn: 2.6575756	total: 9.36s	remaining: 4.26s
    687:	learn: 2.6563032	total: 9.39s	remaining: 4.26s
    688:	learn: 2.6560638	total: 9.4s	remaining: 4.24s
    689:	learn: 2.6550384	total: 9.41s	remaining: 4.23s
    690:	learn: 2.6543462	total: 9.43s	remaining: 4.22s
    691:	learn: 2.6538410	total: 9.44s	remaining: 4.2s
    692:	learn: 2.6529204	total: 9.46s	remaining: 4.19s
    693:	learn: 2.6519810	total: 9.47s	remaining: 4.17s
    694:	learn: 2.6513560	total: 9.48s	remaining: 4.16s
    695:	learn: 2.6504249	total: 9.49s	remaining: 4.14s
    696:	learn: 2.6492246	total: 9.51s	remaining: 4.13s
    697:	learn: 2.6488112	total: 9.52s	remaining: 4.12s
    698:	learn: 2.6477225	total: 9.54s	remaining: 4.11s
    699:	learn: 2.6467550	total: 9.55s	remaining: 4.09s
    700:	learn: 2.6458588	total: 9.56s	remaining: 4.08s
    701:	learn: 2.6454691	total: 9.57s	remaining: 4.06s
    702:	learn: 2.6449218	total: 9.59s	remaining: 4.05s
    703:	learn: 2.6443796	total: 9.6s	remaining: 4.04s
    704:	learn: 2.6429573	total: 9.61s	remaining: 4.02s
    705:	learn: 2.6420455	total: 9.63s	remaining: 4.01s
    706:	learn: 2.6410061	total: 9.65s	remaining: 4s
    707:	learn: 2.6401694	total: 9.66s	remaining: 3.98s
    708:	learn: 2.6386249	total: 9.68s	remaining: 3.97s
    709:	learn: 2.6377049	total: 9.69s	remaining: 3.96s
    710:	learn: 2.6369203	total: 9.71s	remaining: 3.95s
    711:	learn: 2.6360742	total: 9.73s	remaining: 3.93s
    712:	learn: 2.6356195	total: 9.77s	remaining: 3.93s
    713:	learn: 2.6351666	total: 9.78s	remaining: 3.92s
    714:	learn: 2.6344092	total: 9.79s	remaining: 3.9s
    715:	learn: 2.6337397	total: 9.81s	remaining: 3.89s
    716:	learn: 2.6329491	total: 9.82s	remaining: 3.88s
    717:	learn: 2.6323111	total: 9.83s	remaining: 3.86s
    718:	learn: 2.6309827	total: 9.84s	remaining: 3.85s
    719:	learn: 2.6304894	total: 9.86s	remaining: 3.83s
    720:	learn: 2.6287535	total: 9.87s	remaining: 3.82s
    721:	learn: 2.6279561	total: 9.88s	remaining: 3.8s
    722:	learn: 2.6269711	total: 9.89s	remaining: 3.79s
    723:	learn: 2.6262380	total: 9.9s	remaining: 3.77s
    724:	learn: 2.6255719	total: 9.92s	remaining: 3.76s
    725:	learn: 2.6241811	total: 9.93s	remaining: 3.75s
    726:	learn: 2.6233153	total: 9.94s	remaining: 3.73s
    727:	learn: 2.6231162	total: 9.96s	remaining: 3.72s
    728:	learn: 2.6219582	total: 9.97s	remaining: 3.71s
    729:	learn: 2.6217281	total: 9.98s	remaining: 3.69s
    730:	learn: 2.6202994	total: 9.99s	remaining: 3.68s
    731:	learn: 2.6189966	total: 10s	remaining: 3.66s
    732:	learn: 2.6184782	total: 10s	remaining: 3.65s
    733:	learn: 2.6178080	total: 10s	remaining: 3.64s
    734:	learn: 2.6174260	total: 10s	remaining: 3.62s
    735:	learn: 2.6169643	total: 10.1s	remaining: 3.61s
    736:	learn: 2.6161507	total: 10.1s	remaining: 3.6s
    737:	learn: 2.6141453	total: 10.1s	remaining: 3.58s
    738:	learn: 2.6133486	total: 10.1s	remaining: 3.57s
    739:	learn: 2.6125936	total: 10.1s	remaining: 3.56s
    740:	learn: 2.6124634	total: 10.1s	remaining: 3.54s
    741:	learn: 2.6117434	total: 10.2s	remaining: 3.53s
    742:	learn: 2.6111617	total: 10.2s	remaining: 3.52s
    743:	learn: 2.6108941	total: 10.2s	remaining: 3.5s
    744:	learn: 2.6108279	total: 10.2s	remaining: 3.49s
    745:	learn: 2.6104550	total: 10.2s	remaining: 3.47s
    746:	learn: 2.6095009	total: 10.2s	remaining: 3.46s
    747:	learn: 2.6090024	total: 10.2s	remaining: 3.45s
    748:	learn: 2.6081122	total: 10.2s	remaining: 3.43s
    749:	learn: 2.6075409	total: 10.3s	remaining: 3.42s
    750:	learn: 2.6072880	total: 10.3s	remaining: 3.41s
    751:	learn: 2.6061584	total: 10.3s	remaining: 3.4s
    752:	learn: 2.6054910	total: 10.3s	remaining: 3.38s
    753:	learn: 2.6054563	total: 10.3s	remaining: 3.37s
    754:	learn: 2.6047003	total: 10.3s	remaining: 3.35s
    755:	learn: 2.6041005	total: 10.4s	remaining: 3.34s
    756:	learn: 2.6036218	total: 10.4s	remaining: 3.33s
    757:	learn: 2.6029017	total: 10.4s	remaining: 3.32s
    758:	learn: 2.6024385	total: 10.4s	remaining: 3.3s
    759:	learn: 2.6000938	total: 10.4s	remaining: 3.29s
    760:	learn: 2.5995939	total: 10.4s	remaining: 3.28s
    761:	learn: 2.5991995	total: 10.5s	remaining: 3.26s
    762:	learn: 2.5980884	total: 10.5s	remaining: 3.25s
    763:	learn: 2.5969495	total: 10.5s	remaining: 3.23s
    764:	learn: 2.5956175	total: 10.5s	remaining: 3.22s
    765:	learn: 2.5952170	total: 10.5s	remaining: 3.21s
    766:	learn: 2.5947443	total: 10.5s	remaining: 3.19s
    767:	learn: 2.5936144	total: 10.5s	remaining: 3.18s
    768:	learn: 2.5933305	total: 10.5s	remaining: 3.17s
    769:	learn: 2.5924847	total: 10.5s	remaining: 3.15s
    770:	learn: 2.5920933	total: 10.6s	remaining: 3.14s
    771:	learn: 2.5913741	total: 10.6s	remaining: 3.12s
    772:	learn: 2.5904191	total: 10.6s	remaining: 3.11s
    773:	learn: 2.5891834	total: 10.6s	remaining: 3.09s
    774:	learn: 2.5878693	total: 10.6s	remaining: 3.08s
    775:	learn: 2.5867381	total: 10.6s	remaining: 3.07s
    776:	learn: 2.5863358	total: 10.6s	remaining: 3.05s
    777:	learn: 2.5848478	total: 10.6s	remaining: 3.04s
    778:	learn: 2.5834031	total: 10.7s	remaining: 3.03s
    779:	learn: 2.5822201	total: 10.7s	remaining: 3.01s
    780:	learn: 2.5819477	total: 10.7s	remaining: 3s
    781:	learn: 2.5806441	total: 10.7s	remaining: 2.98s
    782:	learn: 2.5797272	total: 10.7s	remaining: 2.97s
    783:	learn: 2.5794316	total: 10.7s	remaining: 2.96s
    784:	learn: 2.5793565	total: 10.7s	remaining: 2.94s
    785:	learn: 2.5789491	total: 10.7s	remaining: 2.93s
    786:	learn: 2.5786130	total: 10.8s	remaining: 2.91s
    787:	learn: 2.5778040	total: 10.8s	remaining: 2.9s
    788:	learn: 2.5771038	total: 10.8s	remaining: 2.88s
    789:	learn: 2.5765422	total: 10.8s	remaining: 2.87s
    790:	learn: 2.5765305	total: 10.8s	remaining: 2.85s
    791:	learn: 2.5759446	total: 10.8s	remaining: 2.84s
    792:	learn: 2.5753319	total: 10.8s	remaining: 2.83s
    793:	learn: 2.5740754	total: 10.9s	remaining: 2.82s
    794:	learn: 2.5715003	total: 10.9s	remaining: 2.8s
    795:	learn: 2.5710514	total: 10.9s	remaining: 2.79s
    796:	learn: 2.5705753	total: 10.9s	remaining: 2.77s
    797:	learn: 2.5694177	total: 10.9s	remaining: 2.76s
    798:	learn: 2.5693288	total: 10.9s	remaining: 2.75s
    799:	learn: 2.5684949	total: 10.9s	remaining: 2.73s
    800:	learn: 2.5681421	total: 10.9s	remaining: 2.72s
    801:	learn: 2.5677420	total: 11s	remaining: 2.71s
    802:	learn: 2.5668554	total: 11s	remaining: 2.69s
    803:	learn: 2.5661610	total: 11s	remaining: 2.68s
    804:	learn: 2.5649298	total: 11s	remaining: 2.67s
    805:	learn: 2.5643757	total: 11s	remaining: 2.65s
    806:	learn: 2.5635926	total: 11s	remaining: 2.64s
    807:	learn: 2.5625960	total: 11.1s	remaining: 2.63s
    808:	learn: 2.5621458	total: 11.1s	remaining: 2.61s
    809:	learn: 2.5611511	total: 11.1s	remaining: 2.6s
    810:	learn: 2.5598865	total: 11.1s	remaining: 2.58s
    811:	learn: 2.5595026	total: 11.1s	remaining: 2.57s
    812:	learn: 2.5590677	total: 11.1s	remaining: 2.56s
    813:	learn: 2.5570864	total: 11.1s	remaining: 2.54s
    814:	learn: 2.5562689	total: 11.1s	remaining: 2.53s
    815:	learn: 2.5551224	total: 11.2s	remaining: 2.51s
    816:	learn: 2.5547338	total: 11.2s	remaining: 2.5s
    817:	learn: 2.5534156	total: 11.2s	remaining: 2.49s
    818:	learn: 2.5524151	total: 11.2s	remaining: 2.47s
    819:	learn: 2.5508074	total: 11.2s	remaining: 2.46s
    820:	learn: 2.5507462	total: 11.2s	remaining: 2.45s
    821:	learn: 2.5505430	total: 11.2s	remaining: 2.43s
    822:	learn: 2.5501027	total: 11.3s	remaining: 2.42s
    823:	learn: 2.5484985	total: 11.3s	remaining: 2.4s
    824:	learn: 2.5466702	total: 11.3s	remaining: 2.39s
    825:	learn: 2.5464747	total: 11.3s	remaining: 2.38s
    826:	learn: 2.5455754	total: 11.3s	remaining: 2.36s
    827:	learn: 2.5450838	total: 11.3s	remaining: 2.35s
    828:	learn: 2.5441940	total: 11.3s	remaining: 2.34s
    829:	learn: 2.5432401	total: 11.4s	remaining: 2.33s
    830:	learn: 2.5418683	total: 11.4s	remaining: 2.31s
    831:	learn: 2.5408294	total: 11.4s	remaining: 2.3s
    832:	learn: 2.5396816	total: 11.4s	remaining: 2.28s
    833:	learn: 2.5388226	total: 11.4s	remaining: 2.27s
    834:	learn: 2.5378257	total: 11.4s	remaining: 2.26s
    835:	learn: 2.5370958	total: 11.4s	remaining: 2.25s
    836:	learn: 2.5367997	total: 11.5s	remaining: 2.23s
    837:	learn: 2.5352276	total: 11.5s	remaining: 2.22s
    838:	learn: 2.5346884	total: 11.5s	remaining: 2.21s
    839:	learn: 2.5343733	total: 11.5s	remaining: 2.19s
    840:	learn: 2.5331365	total: 11.5s	remaining: 2.18s
    841:	learn: 2.5319709	total: 11.5s	remaining: 2.16s
    842:	learn: 2.5302880	total: 11.5s	remaining: 2.15s
    843:	learn: 2.5295666	total: 11.5s	remaining: 2.13s
    844:	learn: 2.5289543	total: 11.6s	remaining: 2.12s
    845:	learn: 2.5282141	total: 11.6s	remaining: 2.11s
    846:	learn: 2.5275829	total: 11.6s	remaining: 2.09s
    847:	learn: 2.5255226	total: 11.6s	remaining: 2.08s
    848:	learn: 2.5247840	total: 11.6s	remaining: 2.06s
    849:	learn: 2.5238768	total: 11.6s	remaining: 2.05s
    850:	learn: 2.5228889	total: 11.6s	remaining: 2.03s
    851:	learn: 2.5219806	total: 11.6s	remaining: 2.02s
    852:	learn: 2.5210722	total: 11.7s	remaining: 2.01s
    853:	learn: 2.5203851	total: 11.7s	remaining: 2s
    854:	learn: 2.5198727	total: 11.7s	remaining: 1.98s
    855:	learn: 2.5185309	total: 11.7s	remaining: 1.97s
    856:	learn: 2.5180022	total: 11.7s	remaining: 1.96s
    857:	learn: 2.5169538	total: 11.7s	remaining: 1.94s
    858:	learn: 2.5161820	total: 11.8s	remaining: 1.93s
    859:	learn: 2.5150311	total: 11.8s	remaining: 1.91s
    860:	learn: 2.5133511	total: 11.8s	remaining: 1.9s
    861:	learn: 2.5119711	total: 11.8s	remaining: 1.89s
    862:	learn: 2.5117133	total: 11.8s	remaining: 1.87s
    863:	learn: 2.5097917	total: 11.8s	remaining: 1.86s
    864:	learn: 2.5081293	total: 11.8s	remaining: 1.84s
    865:	learn: 2.5074507	total: 11.8s	remaining: 1.83s
    866:	learn: 2.5073099	total: 11.8s	remaining: 1.81s
    867:	learn: 2.5068299	total: 11.8s	remaining: 1.8s
    868:	learn: 2.5047688	total: 11.9s	remaining: 1.79s
    869:	learn: 2.5040270	total: 11.9s	remaining: 1.77s
    870:	learn: 2.5033262	total: 11.9s	remaining: 1.76s
    871:	learn: 2.5020489	total: 11.9s	remaining: 1.75s
    872:	learn: 2.5010296	total: 11.9s	remaining: 1.73s
    873:	learn: 2.5004883	total: 11.9s	remaining: 1.72s
    874:	learn: 2.4990075	total: 11.9s	remaining: 1.7s
    875:	learn: 2.4976880	total: 11.9s	remaining: 1.69s
    876:	learn: 2.4966121	total: 11.9s	remaining: 1.68s
    877:	learn: 2.4956597	total: 12s	remaining: 1.66s
    878:	learn: 2.4945212	total: 12s	remaining: 1.65s
    879:	learn: 2.4939006	total: 12s	remaining: 1.64s
    880:	learn: 2.4930777	total: 12s	remaining: 1.63s
    881:	learn: 2.4917491	total: 12s	remaining: 1.61s
    882:	learn: 2.4909053	total: 12.1s	remaining: 1.6s
    883:	learn: 2.4904990	total: 12.1s	remaining: 1.58s
    884:	learn: 2.4889481	total: 12.1s	remaining: 1.57s
    885:	learn: 2.4883434	total: 12.1s	remaining: 1.56s
    886:	learn: 2.4877706	total: 12.1s	remaining: 1.54s
    887:	learn: 2.4874850	total: 12.1s	remaining: 1.53s
    888:	learn: 2.4870709	total: 12.2s	remaining: 1.52s
    889:	learn: 2.4849084	total: 12.2s	remaining: 1.5s
    890:	learn: 2.4847386	total: 12.2s	remaining: 1.49s
    891:	learn: 2.4825117	total: 12.2s	remaining: 1.48s
    892:	learn: 2.4812147	total: 12.2s	remaining: 1.46s
    893:	learn: 2.4805404	total: 12.2s	remaining: 1.45s
    894:	learn: 2.4795801	total: 12.2s	remaining: 1.44s
    895:	learn: 2.4793987	total: 12.3s	remaining: 1.42s
    896:	learn: 2.4777710	total: 12.3s	remaining: 1.41s
    897:	learn: 2.4769254	total: 12.3s	remaining: 1.4s
    898:	learn: 2.4761909	total: 12.3s	remaining: 1.38s
    899:	learn: 2.4752754	total: 12.3s	remaining: 1.37s
    900:	learn: 2.4731699	total: 12.3s	remaining: 1.35s
    901:	learn: 2.4728074	total: 12.3s	remaining: 1.34s
    902:	learn: 2.4723108	total: 12.4s	remaining: 1.33s
    903:	learn: 2.4718616	total: 12.4s	remaining: 1.31s
    904:	learn: 2.4710376	total: 12.4s	remaining: 1.3s
    905:	learn: 2.4700952	total: 12.4s	remaining: 1.29s
    906:	learn: 2.4691732	total: 12.4s	remaining: 1.27s
    907:	learn: 2.4682919	total: 12.4s	remaining: 1.26s
    908:	learn: 2.4673380	total: 12.4s	remaining: 1.24s
    909:	learn: 2.4666571	total: 12.4s	remaining: 1.23s
    910:	learn: 2.4662773	total: 12.5s	remaining: 1.22s
    911:	learn: 2.4658879	total: 12.5s	remaining: 1.2s
    912:	learn: 2.4654825	total: 12.5s	remaining: 1.19s
    913:	learn: 2.4653855	total: 12.5s	remaining: 1.18s
    914:	learn: 2.4646718	total: 12.5s	remaining: 1.16s
    915:	learn: 2.4635841	total: 12.5s	remaining: 1.15s
    916:	learn: 2.4620678	total: 12.6s	remaining: 1.14s
    917:	learn: 2.4617298	total: 12.6s	remaining: 1.12s
    918:	learn: 2.4613498	total: 12.6s	remaining: 1.11s
    919:	learn: 2.4606236	total: 12.6s	remaining: 1.09s
    920:	learn: 2.4597137	total: 12.6s	remaining: 1.08s
    921:	learn: 2.4593069	total: 12.6s	remaining: 1.07s
    922:	learn: 2.4582584	total: 12.6s	remaining: 1.05s
    923:	learn: 2.4577198	total: 12.7s	remaining: 1.04s
    924:	learn: 2.4573976	total: 12.7s	remaining: 1.03s
    925:	learn: 2.4568140	total: 12.7s	remaining: 1.01s
    926:	learn: 2.4562306	total: 12.7s	remaining: 1s
    927:	learn: 2.4558771	total: 12.7s	remaining: 986ms
    928:	learn: 2.4554482	total: 12.7s	remaining: 972ms
    929:	learn: 2.4549644	total: 12.7s	remaining: 958ms
    930:	learn: 2.4539255	total: 12.7s	remaining: 945ms
    931:	learn: 2.4533592	total: 12.8s	remaining: 931ms
    932:	learn: 2.4527900	total: 12.8s	remaining: 918ms
    933:	learn: 2.4523352	total: 12.8s	remaining: 904ms
    934:	learn: 2.4509713	total: 12.8s	remaining: 890ms
    935:	learn: 2.4499706	total: 12.8s	remaining: 877ms
    936:	learn: 2.4491672	total: 12.8s	remaining: 863ms
    937:	learn: 2.4488545	total: 12.8s	remaining: 849ms
    938:	learn: 2.4480655	total: 12.9s	remaining: 835ms
    939:	learn: 2.4478545	total: 12.9s	remaining: 821ms
    940:	learn: 2.4468212	total: 12.9s	remaining: 808ms
    941:	learn: 2.4461211	total: 12.9s	remaining: 794ms
    942:	learn: 2.4449037	total: 12.9s	remaining: 780ms
    943:	learn: 2.4440311	total: 12.9s	remaining: 767ms
    944:	learn: 2.4436324	total: 12.9s	remaining: 753ms
    945:	learn: 2.4431516	total: 13s	remaining: 739ms
    946:	learn: 2.4412523	total: 13s	remaining: 726ms
    947:	learn: 2.4407005	total: 13s	remaining: 712ms
    948:	learn: 2.4403728	total: 13s	remaining: 699ms
    949:	learn: 2.4393770	total: 13s	remaining: 685ms
    950:	learn: 2.4387986	total: 13s	remaining: 671ms
    951:	learn: 2.4381110	total: 13s	remaining: 657ms
    952:	learn: 2.4365917	total: 13.1s	remaining: 644ms
    953:	learn: 2.4361810	total: 13.1s	remaining: 630ms
    954:	learn: 2.4357174	total: 13.1s	remaining: 616ms
    955:	learn: 2.4336812	total: 13.1s	remaining: 602ms
    956:	learn: 2.4329775	total: 13.1s	remaining: 589ms
    957:	learn: 2.4315028	total: 13.1s	remaining: 575ms
    958:	learn: 2.4311447	total: 13.1s	remaining: 561ms
    959:	learn: 2.4307185	total: 13.1s	remaining: 547ms
    960:	learn: 2.4295355	total: 13.1s	remaining: 533ms
    961:	learn: 2.4288674	total: 13.2s	remaining: 520ms
    962:	learn: 2.4283197	total: 13.2s	remaining: 506ms
    963:	learn: 2.4271379	total: 13.2s	remaining: 492ms
    964:	learn: 2.4265305	total: 13.2s	remaining: 478ms
    965:	learn: 2.4251540	total: 13.2s	remaining: 465ms
    966:	learn: 2.4245170	total: 13.2s	remaining: 452ms
    967:	learn: 2.4241233	total: 13.3s	remaining: 438ms
    968:	learn: 2.4228552	total: 13.3s	remaining: 424ms
    969:	learn: 2.4217854	total: 13.3s	remaining: 410ms
    970:	learn: 2.4212783	total: 13.3s	remaining: 397ms
    971:	learn: 2.4203148	total: 13.3s	remaining: 383ms
    972:	learn: 2.4200807	total: 13.3s	remaining: 369ms
    973:	learn: 2.4195727	total: 13.3s	remaining: 356ms
    974:	learn: 2.4185728	total: 13.3s	remaining: 342ms
    975:	learn: 2.4174205	total: 13.4s	remaining: 328ms
    976:	learn: 2.4165883	total: 13.4s	remaining: 315ms
    977:	learn: 2.4160089	total: 13.4s	remaining: 301ms
    978:	learn: 2.4146466	total: 13.4s	remaining: 288ms
    979:	learn: 2.4138030	total: 13.4s	remaining: 274ms
    980:	learn: 2.4134217	total: 13.4s	remaining: 260ms
    981:	learn: 2.4125268	total: 13.5s	remaining: 247ms
    982:	learn: 2.4111877	total: 13.5s	remaining: 233ms
    983:	learn: 2.4110279	total: 13.5s	remaining: 219ms
    984:	learn: 2.4104205	total: 13.5s	remaining: 206ms
    985:	learn: 2.4099553	total: 13.5s	remaining: 192ms
    986:	learn: 2.4099318	total: 13.5s	remaining: 178ms
    987:	learn: 2.4096158	total: 13.5s	remaining: 164ms
    988:	learn: 2.4083957	total: 13.6s	remaining: 151ms
    989:	learn: 2.4071405	total: 13.6s	remaining: 137ms
    990:	learn: 2.4065814	total: 13.6s	remaining: 123ms
    991:	learn: 2.4056082	total: 13.6s	remaining: 110ms
    992:	learn: 2.4054898	total: 13.6s	remaining: 95.9ms
    993:	learn: 2.4050936	total: 13.6s	remaining: 82.2ms
    994:	learn: 2.4042304	total: 13.6s	remaining: 68.5ms
    995:	learn: 2.4035576	total: 13.7s	remaining: 54.9ms
    996:	learn: 2.4032842	total: 13.7s	remaining: 41.2ms
    997:	learn: 2.4024237	total: 13.7s	remaining: 27.5ms
    998:	learn: 2.4018406	total: 13.7s	remaining: 13.7ms
    999:	learn: 2.4001612	total: 13.7s	remaining: 0us
    




    <catboost.core.CatBoostRegressor at 0x14dd826b550>




```python
merged_right = pd.merge(left=df, right=test, how='right', left_on='Emp_ID', right_on='Emp_ID')
merged_right
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
      <td>394</td>
      <td>32</td>
      <td>Female</td>
      <td>C20</td>
      <td>Master</td>
      <td>97722</td>
      <td>24</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>173</td>
      <td>37</td>
      <td>Male</td>
      <td>C28</td>
      <td>College</td>
      <td>56174</td>
      <td>24</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1090</td>
      <td>37</td>
      <td>Male</td>
      <td>C13</td>
      <td>College</td>
      <td>96750</td>
      <td>24</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>840</td>
      <td>39</td>
      <td>Female</td>
      <td>C8</td>
      <td>College</td>
      <td>88813</td>
      <td>24</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>308</td>
      <td>30</td>
      <td>Male</td>
      <td>C5</td>
      <td>Master</td>
      <td>188418</td>
      <td>24</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
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
      <th>736</th>
      <td>2134</td>
      <td>38</td>
      <td>Male</td>
      <td>C29</td>
      <td>College</td>
      <td>116006</td>
      <td>24</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>737</th>
      <td>2255</td>
      <td>38</td>
      <td>Male</td>
      <td>C25</td>
      <td>College</td>
      <td>133489</td>
      <td>24</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>738</th>
      <td>448</td>
      <td>35</td>
      <td>Male</td>
      <td>C10</td>
      <td>Bachelor</td>
      <td>65389</td>
      <td>24</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>739</th>
      <td>1644</td>
      <td>46</td>
      <td>Female</td>
      <td>C9</td>
      <td>Bachelor</td>
      <td>105513</td>
      <td>24</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>740</th>
      <td>624</td>
      <td>33</td>
      <td>Male</td>
      <td>C15</td>
      <td>Bachelor</td>
      <td>104712</td>
      <td>24</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>741 rows × 11 columns</p>
</div>




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
