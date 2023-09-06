#!/bin/sh


# Download datasets in datasets/ folder if they are not present
# datasets are named english.200MB, english.1024MB, proteins.200MB

datasets_folder="datasets"
nlang_datasets="english.200MB english.1024MB"
# check if they exist in datasets folder
mkdir -p $datasets_folder
for dataset in $nlang_datasets
do
    if [ ! -f "$datasets_folder/$dataset" ]; then
        echo "Downloading $dataset"
        wget http://pizzachili.dcc.uchile.cl/texts/nlang/$dataset.gz -O datasets/$dataset.gz
        gunzip datasets/$dataset.gz
    fi
done

proteins_datasets="proteins.200MB proteins.50MB"

for dataset in $proteins_datasets
do
    if [ ! -f "$datasets_folder/$dataset" ]; then
        echo "Downloading $dataset"
        wget http://pizzachili.dcc.uchile.cl/texts/protein/$dataset.gz -O datasets/$dataset.gz
        gunzip datasets/$dataset.gz
    fi
done

repcorpus_real_datasets="einstein.en.txt"

for dataset in $repcorpus_real_datasets
do
    if [ ! -f "$datasets_folder/$dataset" ]; then
        echo "Downloading $dataset"
        wget http://pizzachili.dcc.uchile.cl/repcorpus/real/$dataset.gz -O datasets/$dataset.gz
        gunzip datasets/$dataset.gz
    fi
done

RUSTFLAGS="-C target-cpu=native" cargo run --release && \
cd sdsl && \
make && \
./benchmarks.out && \
cd .. && \
cd visualization && \
./generate_figures.py && \
cd ..