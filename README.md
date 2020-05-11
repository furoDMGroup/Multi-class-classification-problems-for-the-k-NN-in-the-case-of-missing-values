# Multi-class classification problems for the k-NN in the case of missing values

## Paper
This repository contains the implementation of algorithms and experiments that were  
made in the following paper:
```
U. Bentkowska, J. G. Bazan, M. Mrukowicz, L. Zaręba, P. Molenda,
Multi-class classification problems for the k-NN algorithm in the case of missing values, 
FUZZ-IEEE 2020 submitted
```
Please cite us if you use this source code or the results of this project in your research.

## Implemented algorithms
This repository contains the implementation of two extensions of
the classic KNN algorithm. The project is intended to be compatible with [scikit-learn](https://scikit-learn.org/stable/)
and is using its conventions. [NumPy](https://numpy.org/) is used to implement low-level operations, while
[pandas](https://pandas.pydata.org/) is used mainly for IO purposes.

Implemented algorithms are:\
Algorithm F, which is using Interval-Valued Fuzzy Sets concepts to deal with missing values, especially  
aggregation functions, \
Algorithm M, which deals with missing values like WEKA k-NN implementation. It should be considered as  
comparative implementation (control subject) in research.

The both F and M algorithms are binary classifiers, however they are extended to
the multi-class version, based on scikit-learn's OneVsRest class. In the case of
F classifier, there was a need to modify this class. Since random insert of missing data in each binary algorithm F is independent,
there  is  a  possibility,  especially  in  small datasets and for small k that testing
object  will  be  "moved" to different training objects each time (perhaps with different decision).
Due to this fact, it is possible that multi-class MF decision vector will be a vector of zeros.
In such case, which may be considered a tie, we give all classes equal probability and choose class with maximum
value, which is simply the first decision class. This approach gives sufficient classification quality results, however more approaches
should be proposed and studied in the future.

The multi-class versions of algorithms are called MF and MM in the paper.

## Reproduce results

### Download source code
The preffered way to use this source code is to clone this repository:
```
git clone https://github.com/furoDMGroup/Multi-class-classification-problems-for-the-k-NN-in-the-case-of-missing-values
```

### Prerequisites
You should have a Python interpreter installed. This repository was tested mainly for
Python 3.7, both 32 and 64-bit versions.
You should have installed the following packages, listed in [requirements.txt](requirements.txt): numpy, scipy, pandas, scikit-learn, xlrd, requests.
To install them with pip use the following command in the project root folder:
```
pip install -r requirements.txt
```

### Preparation of datasets
First please specify destination path for datasets in [setup_path.py](reproduce_results/setup_path.py) script, by changing the value of datasets_path variable.
Make sure that the given path is accessible by standard operating system user or run all scripts in privileged mode.
To download datasets, please execute [download_datasets.py](reproduce_results/download_datasets.py) script. This script will download
datasets from the UCI repository and make all necessary steps to prepare them for the learning process (including unzipping files from archive).

### Reproduce each dataset result
In reproduce_results package there are scripts with names corresponding to each dataset from
UCI. To reproduce results please execute those scripts. Results will be saved into spreadsheet files.
To sort results, according to the level of missing values and AUC values you could use [sort_result.py](reproduce_results/sort_result.py)
script by passing the file name as command line argument:
```
sort_result.py input_file.xlsx
```
The sorted file will be named: sorted_input_file.xlsx

## Author
If you have any issues with source code or have any question please contact with:\
[Marcin Mrukowicz](https://github.com/MarcinMrukowicz)

## License

### This project is licensed under BSD3 Licence - see the [LICENSE](LICENSE) file for details.