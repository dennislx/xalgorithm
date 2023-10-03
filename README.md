My Implementation of Data Structure and Algorithms in Python
============================================================

## Build and Update Package

```
make upload version=a.b.c
```

## Install This Package

```
pip install xalgorithm
```

## List of Implementations

| Xalgorithm Function   | Effect |
| ----------- | ------------------------------------------ |
| `@tag_me`    | Tag each function with a label and access the function using that label |
| `@print_me`  | Print the functon, its argument and result, helpful in debugging |
| `@ctrl_c`    | Ask the user for confirmation to exit the program |
| `@record_it` | Measure either the execution time or the execution count of a function |
| `opath`     | Return the absolute path, expanding any user shortcuts |
| `ojoin`     | Join multiple path components and optionally creates the path if it doesn't exist |
| `osimplify` | Normalize a path |
| `xprint`    | Print text in styled format, please see my changelog for detailed usages |

| Out Functon | Effect |
| ----------- | ------------------------------------------ |
| `make_classification`  |  Generates a synthetic classification dataset | 
| `describe_df` | Performs exploratory analysis on each feature |
| `print_df` | Print a pandas DataFrame with rich API |
| `Benchmark`| This class evaluates several well-known machine learning algorithms for a classification problem and provides multiple classification metrics for comparative analysis  | 
| `%py_version` | Print the current version of Python |
| `%%csv`    |  Parses the cell into a DataFrame and then prints the DataFrame to the console in the specified format (rich-format, plain-format, markdown-format) |
| `%%time`   |  Display the execution time of code within a cell in seconds |

| Ods Function | Effect |
| ----------- | ------------------------------------------ |
| `power_analysis` | Calculate sample size for power analysis using effect size, significance, power, and population std |
| `RFM` | Segment and Target customers for personalized marketing strategies using Recency, Frequency, Monetary analysis |
| `parse_vds` | Download Youtube subtitles, see `xalgorithm --help` for guidance. The Spacy English pipeline can be optionally loaded to segment sentences within subtitle scripts  |





| Array Algorithm  | Definition |
| ----------- | ------------------------------------------ |
| :white_large_square: Sliding Window   |  |
| :white_check_mark: HuaRongDao  | Solve a puzzle where the goal is to start from initial state and end in final state |


## Advanced Algorithms and Data Structures (La Rocca, M., 2021)

| Topic  | Note | Finish Date |
| :---------------  | :----------------- | :---- |
| Chp 2: d-way heaps | Priority Queue | :white_large_square: |
| Chp 2: huffman compression | xxxx | :white_large_square: |

## ChangeLog

All updates are tracked in [this file](./CHANGELOG.md). 


## Credits


To better prepare myself for the challenges of algorithmic problems, I have included code from [Open Data Structures](https://opendatastructures.org/). I will probably to rewrite some functions to make them compatible with my project as I continue to read this book. I owe huge gratitude for their work in jumping start my learning journey.