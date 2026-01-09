#!/bin/bash


### Argument parser
while getopts o:s: arg
do
    case "${arg}" in
        o) # Specify output directory for orthologs fasta files
            outdir=${OPTARG};;
        s)  # Specify target species
            species=${OPTARG};;
    esac
done

### download fasta file of orthologous sequences from OrthoDB
echo "Downloading fasta files of orthologs from OrthoDB."

while IFS=$',' read -r -a myArray
do
    uniprot_id="${myArray[0]}"
    orthodb_id="${myArray[1]}"
    # level=2 and species=2 set the orthology level to bacteria
    # if none, orthologs are searched within all species
    url="https://v101.orthodb.org/fasta?query=${orthodb_id}" #&level=2&species=2"
    filename="${outdir}/${uniprot_id}_orthologs.fasta"
    wget "$url" -O "$filename"
done < "${species}_ids.txt"

