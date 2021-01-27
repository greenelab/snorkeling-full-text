# Snorkeling-Full-Text

This is an upgraded version of the original [snorkeling project](https://github.com/greenelab/snorkeling).
The goal here is to see if full text will improve distant supervision performance.

## Before You Begin

Please make sure you have executed the pubtator repository first as the code in the notebook requires pubtator to have been executed.
First run `git clone https://github.com/greenelab/pubtator` in the same directory that will hold this repository, then follow the instructions that are located in [Pubtator's repository](https://github.com/greenelab/pubtator).

## Installation Instructions

Snorkeling-full-text uses [conda](http://conda.pydata.org/docs/intro.html) as a python package manager.
Before moving on to the instructions below, please make sure to have it installed.
[Download conda here!!](https://docs.conda.io/en/latest/miniconda.html)

Once everything has been installed, type following command in the terminal:

```bash
bash install.sh
```
_Note_:
There is a bash command within the install.sh that only works on unix systems.
If you are on windows (and possibly mac), you should remove that file or execute each command individually.
Alternatively for windows users please refer to these [installation instructions](https://itsfoss.com/install-bash-on-windows/) to enable bash on windows.

You can activate the environment by using the following command:

```bash
conda activate snorkeling_full_text
```

## License

Please look at [LICENSE.md](LICENSE.md).
