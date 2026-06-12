from pyfaidx import Fasta
from evolution.extract_sequences import run_extraction
import pandas as pd

CONSTRUCT_PATH = "src/workflows/overlap_analysis/correct_construct/pPSntF_enhLib_35Spr.fa"
BARCODE_PATH = "src/workflows/overlap_analysis/correct_construct/barcodeX.fa"
FASTA_OUT = "src/workflows/overlap_analysis/correct_construct/construct_X.fa"
STARRSEQ_PATH = "src/workflows/overlap_analysis/data/plantstarr-seq_main_light_and_dark_simon_gernot.csv"
ANNOTATION_PATH = "src/workflows/overlap_analysis/correct_construct/construct_annotation.gtf"
CONSTRUCT_INSERT_PATH = "src/workflows/overlap_analysis/correct_construct/construct_inserts.fa"

def create_fastas():
    construct_seq = str(Fasta(CONSTRUCT_PATH)[0])
    barcode_fastas = [Fasta(BARCODE_PATH.replace("X", str(i))) for i in range(1, 3)]
    barcodes = [str(fasta[0]) for fasta in barcode_fastas]

    new_constructs = [construct_seq.replace("VNNVNNVNNVNNVNNVNN", barcode) for barcode in barcodes]
    front_extended_constructs = [construct[-1000:] + construct for construct in new_constructs]
    for i, construct in enumerate(front_extended_constructs):
        with open(FASTA_OUT.replace("X", str(i+1)), "w") as f:
            f.write(f">1\n{construct}\n")


def extract_sequences_here():
    for i in range(1, 3):
        run_extraction(
            fasta_path=FASTA_OUT.replace("X", str(i)),
            annotation_path=ANNOTATION_PATH,
            output_path=FASTA_OUT.replace("X", str(i)).replace(".fa", "_extracted.fa"),
            intragenic=500,
            extragenic=1000,
            gene_name_attribute="gene_id",
            feature_type_filter=["gene"],
            genes_of_interest=[],
            vcf_paths=[]
        )


def insert_fragments():
    # load the starr-seq data and filter to the relevant constructs
    starrseq_data = pd.read_csv(STARRSEQ_PATH)
    relevant_columns = ["id", "sequence"]
    starrseq_data = starrseq_data[relevant_columns].drop_duplicates()
    starrseq_data = starrseq_data[starrseq_data["id"].str.startswith("bHLH_") | starrseq_data["id"].str.startswith("WRKY_")]

    out_string = ""
    # load fastas
    for i in range(1, 3):
        fasta_path = FASTA_OUT.replace("X", str(i)).replace(".fa", "_extracted.fa")
        background = str(Fasta(fasta_path)[0])
        print("background length", len(background))
        print("background N-count", background.count("N"))
        for _, row in starrseq_data.iterrows():
            id_ = row["id"]
            insert = row["sequence"]
            curr_variant = background.replace("N" * 170, insert)
            out_string += f">barcode_{i}_{id_}\n{curr_variant}\n"
            n_count = curr_variant.count('N')
            if n_count > 20:
                print(f"Processed {id_} for barcode {i}")
                print(f"Current variant length: {len(curr_variant)}")
                print(f"Current variant N-count: {n_count}\n")
    with open(construct_insert_path, "w") as f:
        f.write(out_string)


if __name__ == "__main__":
    # create_fastas()
    # extract_sequences_here()
    insert_fragments()