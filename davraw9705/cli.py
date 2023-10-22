import argparse


def get_args():
    """Get args from CLI input"""
    parser = argparse.ArgumentParser(description="DAVRAW9705 - Conversion",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional args
    parser.add_argument('source_dir', type=str,
                        help="Dir with raw and aux files.")
    parser.add_argument('output_dir', type=str,
                        help="Dir for generated output.")

    args = parser.parse_args()
    return args
