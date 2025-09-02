import argparse
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from function.calculate_similarity_edited import calc_similarity
import torch


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Parse unknown variables for the script.")

    # Add arguments for all variables
    parser.add_argument('--image_dir', type=str, required=True, 
                        help="Directory containing the image data.")
    parser.add_argument('--waterbird_csv', type=str, required=True, 
                        help="Filename for the waterbird CSV file.")
    parser.add_argument('--result_csv', type=str, required=True, 
                        help="Filename for the waterbird result CSV file.")
    parser.add_argument('--save_csv', type=str, required=True, 
                        help="Path where output (similarities csv) will be saved.")
    parser.add_argument('--drive_path', type=str, required=False, default='', 
                    help="Path to the drive directory containing CSV files.")
    
    args = parser.parse_args()

    # Read the CSV files
    cols = ['Keyword', 'Score', 'Acc.', 'Bias']
    df_waterbird = pd.read_csv(args.drive_path + args.waterbird_csv)[cols]

    cols_result = ['image', 'pred', 'actual', 'group', 'spurious', 'correct', 'caption']
    df_waterbird_result = pd.read_csv(args.drive_path + args.result_csv)[cols_result]

    # Extract keywords and calculate similarities
    keywords = df_waterbird["Keyword"].values
    similarities_all = calc_similarity(args.image_dir, df_waterbird_result["image"], keywords)
    new_df = pd.DataFrame(similarities_all.cpu(), index=df_waterbird_result.index,columns=keywords)
    new_df.to_csv(args.save_csv)

if __name__ == "__main__":
    main()
