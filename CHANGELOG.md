## Sep 3th, 2023

Implemented 4 decorators:
    
- print me: useful in debugging, will output function, its argument and result
  <details>

  ```py
  @print_me
  def add(x, y): return x+y 

  >>> add(2, 3) 
  add(2, 3) = 5
  ```
  </details>

- tag me: tag each function with one or multiple labels and we can easily track this function later on with category name and function name
  <details>

  ```py
  @tag_me('dynamic programming')
  @lru_cache()
  def fib(n):
    if n <= 2: return 1
    return fib(n-1) + fib(n-2)
  >>> tag_me('dynamic programming').invoke('fib', 5) # fib(5)
  ```
  </details>

- ctrl+c and record it: The former reminds users to confirm program exit, preventing accidental closure and `record_it` measures function execution time or count and logs the result.
  <details>

  ```py
  @record_it(stat='time', name="timing function")
  @record_it(stat='count', name="count function")
  @ctrl_c
  def calculate_million_numbers(num):
    x = 0
    for _ in range(num): x += 1
  >>> calculate_million_numbers(1_000_000)
  ```
  </details>

## Sep 4th

Implemented a bunch of utility functions, these include

| function | effect                                          |
| -------- | ----------------------------------------------- |
| ojoin    | join multiple path components and optionally creates the path if it does not exist  |
| ocode    | launch a file (full/relative path) in vscode    |
| omake    | create directories for the given file paths if they don't already exist |
| ofind    | serach for files in a given directory that match a specific pattern  |
| oexists  | check if all the give paths exist |
| osplit   | split a given path into two parts based on a specified separator |

## Sep 5th

Implemented several array-based algorithms

| function         | notes                                           |
| --------         | ----------------------------------------------- |
| `freq_lt_ntimes` | Each number in the list can occur no more than n times |
| `flatten_nested_array` | Flatten nested arrays |
| `hua_rong_dao`   | Determine the minimum steps required to rearrange numbers from initial state to the final state |

## Sep 7th

Implemented multiple metrics for ML classification and power analysis for A/B testing. Keep in mind that `P` stands for all real positive cases, and `P'` stands for all predicted positive cases
:fire: TODO implement ROC curves, AUC calculations from scratch

| function         | notes                                           |
| --------         | ----------------------------------------------- |
| accuracy         | (TP + TN) / (P + N)                             |
| precision        | TP / P' (or minimize type I error)              |
| recall           | TP / P  (or minimize type II error)             |
| `f1_score`       | 2 x TP / (P + P')                               |
| `power_analysis` | calculate the sample size based on minimum detectable effect size, significance level, desired power, standard deviation, and other parameters |

## Sep 8th

Implemented RFM analysis, which takes an input of a DataFrame with customer IDs, purchase details (item, cost, and time), and allows you to customize segmentation rules. It outputs customer segments. Additionally, you can create three distribution plots as shown below.

| dist   | score | segment |
| ------ | ------| ------ |
| ![](./figs/dist_of_freq.png) | ![](./figs/count_of_score.png) | ![](./figs/count_of_label.png) |

<details>

```py
df = (
  pd.read_csv(data_path)                                    # read data file
  .pipe(clean_names)                                        # clean column names
  .count_cumulative_unique('customer_name', 'customer_id')  # obtain customer id
  .currency_column_to_numeric("sales")                      # convert currencies
  .rename(columns = {'sales': 'order_amount'})              # rename column
)  
rfm = RFM(df)
```
</details>

## Sep 10th

Implemented the K-nearest neighbor algorithm with a fixed hyperparameter `k`. I haven't included the elbow method for selecting the optimal `k` yet. KNN is a lazy learning algorithm that relies on data for predictions, without constructing a parametric model
:fire: Elbow method for selecting K and KNN visualization

## Sep 16th

Implemented print dataframe function built on top of rich API...

## Sep 20th

Implemented the `benchmark_classification()`, drawing inspiration from [LazyPredict](https://lazypredict.readthedocs.io/en/latest/). The concept behind this function is to measure the difficulty of a tabular classification task by running through a comprehensive set of standard machine learning classifiers, without adopting ensemble methods or fine-tuning hyperparameters. Here is an example of what the output looks like.

![](https://i.imgur.com/LqAIgY1.png)