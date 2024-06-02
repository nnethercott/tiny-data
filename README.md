# tiny-data

A rust-based cli tool for building computer vision datasets built with [reqwest](https://docs.rs/reqwest/latest/reqwest/) and [tokio](https://tokio.rs/).

![alt text](https://github.com/nnethercott/tiny-data/blob/main/assets/images/demo.gif?raw=true)

You can get a list of the available options by running the command below:

```bash
>> tiny-data -h
Usage: tiny-data [OPTIONS]

Options:
  -t, --topics <TOPICS>...   Space-delimited list of image classes
  -n, --nsamples <NSAMPLES>  number of images to download per-class [default: 20]
  -d, --dir <DIR>            name of directory to save to [default: images]
  -h, --help                 Print help
```

Example:

```bash
>> tiny-data --topics bats wombats -n 10 --dir images
>> tree images
images
├── bats
│   ├── 0.jpeg
│   ├── 1.jpeg
│   ├── 2.jpeg
│   ├── 3.jpeg
│   ├── 4.jpeg
│   ├── 5.jpeg
│   ├── 6.jpeg
│   ├── 7.jpeg
│   ├── 8.jpeg
│   └── 9.jpeg
└── wombats
    ├── 0.jpeg
    ├── 1.jpeg
    ├── 2.jpeg
    ├── 3.jpeg
    ├── 4.jpeg
    ├── 5.jpeg
    ├── 6.jpeg
    ├── 7.jpeg
    ├── 8.jpeg
    └── 9.jpeg
```

# Installation

To get started with `tiny-data` you need to enable the [Custom Search API](https://developers.google.com/custom-search/v1/overview) from Google and export the variables `SEARCH_ENGINE_ID` and `CUSTOM_SEARCH_API_KEY` to your environment.

**Note:** google limits the number of requests to 100/day which inherently puts a cap on the number of images you can download.

The package itself can be downloaded from [crates.io](https://crates.io/) by running:

```bash
cargo install tiny-data
```

The python bindings for the package can be downloaded from pypi with additional features for post-download filtering using CLIP by running:

```bash 
pip install tinydata[ml]
```
Make sure you also install the appropriate version of `torch` from [here](https://pytorch.org/get-started/locally/) if you want to use open clip. 
