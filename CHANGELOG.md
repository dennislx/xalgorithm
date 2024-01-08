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

Implemented the `benchmark_classification()`, drawing inspiration from [LazyPredict](https://lazypredict.readthedocs.io/en/latest/). The concept behind this function is to measure the difficulty of a tabular classification task by running through a comprehensive set of standard machine learning classifiers, without adopting ensemble methods or fine-tuning hyperparameters. Here is an example of what the output looks like (A rather imbalanced dataset with positive cases : negative cases = 1 : 9).

![](https://i.imgur.com/1USXL5O.png)


## Oct 1st

Implemented several magic commands in `out/magic_cmd.py`. Initially, my motivation was to record the execution time for certain functions. However, I later discovered that VSCode provides cell execution time by default. 

It allows me to create decorator functions for magic command functions, such as cache management. Indeed, this will prove beneficial on my data science journey, especially when collaborating with other data science colleagues who are more comfortable to use Jupyter Notebook.


```py
%load_ext xalgorithm.out
```
Line Magic: ['%py_version'] 

Cell Magic: ['%%csv', '%%time']

```
%%csv --format rich

tag, input, effects
b, [b]test[/b], boldface text
i, [i]test[/i], display the content in italic style
u, [u]test[/u], underline the text
s, [s]test[/s], draws a horizontal line over the text
dim, [dim]test[/dim], decrease contrast of text
fg, [fg red]test[/fg], color text in red
bg, "[bg 255,0,0]test[/bg]", set background color of text to red
```

![](https://i.imgur.com/faWTU3M.png)

### Oct 2nd

I find that `rich.print` is not user-friendly when it comes to changing font size. The font size is larger than my other text in the Jupyter notebook, so I'd like to implement my own print function. Inspired by [bbcode](https://dcwatson.github.io/bbcode/) and [sty](https://sty.mewo.dev/), I write functions to parse strings with BBCode-style defined styles, including fg(foreground color), bg(background color), and a series of effects (e.g., underline, strike, bold, italic, etc). This way, users can generate text with styles and print it in the terminal or Jupyter console.

```py
xprint("[b]bold[/b] [red]and[/red] [green][u]underline in green[/u][/green]")
```

$$\textbf{bold} \textcolor{red}{\ \text{and}} \underline{\textcolor{green}{\ \text{underline\ in \ green}}}$$


### Oct 3rd

I didn't implemented my own SQL magic `cell/line` command, but I have loaded sql magic from [this work](https://github.com/catherinedevlin/ipython-sql). Although IPython allows you to register multiple classes and functions with a single function call, you need to import/execute `@cell/@line` first before importing scripts that contain `@magics_class`. This is due to its implementation of the decorator: `magics_class`. Once I have registered SqlMagic, I can connect to my local PostgreSQL database running on port 5432 and write SQL queries to interact with the database effectively.

```py
%load_ext xalgorithm.out
conn = 'postgresql://{username}:{password}@localhost/{database}'
%sql $conn
```

```py
%%sql
-- put fetched table to python variable to use it later on
result << 
SELECT actor_id, last_name
FROM actor
LIMIT 5
```

### Oct 7th

Implemented functions to plot distribution of each feature in a tabular data. 

![](./figs/dist_of_shape.png)

