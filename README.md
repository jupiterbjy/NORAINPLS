# NORAINPLS
Not Original, Reckless Attempt In Nasty Pouring Likelihood Suggestion

This is part of Hong-Ik University 'Advanced ML' Lecture project.

## Dir structure

- _data_ : Training data
  - _raw_
    - raw data from [KMA](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do)
  - _preprocessed_
    - Preprocessed raw data w/o sensor anomaly data, split into parts from where sensor anomaly was. Naming sequence is as following:
      `{year}_chunk{chunk_num}` - i.e. `2014_chunk0`, `2015_chunk0`, `2015_chunk1`

---

- _ipynb_ : Demonstration & Documentation purpose
  - `train.ipynb`
    - combination of below 3 scripts in jupyter notebook

---

- `preprocess.py`
  - preprocessing module

- `network.py`
  - network definition

- `train.py`
  - training process
