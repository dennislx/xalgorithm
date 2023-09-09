import argparse
from xalgorithm.ods.vds import parse_vds

def main():
    parser = argparse.ArgumentParser(description="Main Xalgorithm CLI")
    subcmd = parser.add_subparsers(help="Subcommands")
    
    # Define the "subtitle" subcommand
    title_parser = subcmd.add_parser("subtitle", help="Download Youtube Subtitles")
    title_parser.set_defaults(func=parse_vds)
    title_parser.add_argument('video_ids', nargs='+', type=str, help='List of YouTube video IDs.')
    title_parser.add_argument('--spacy', action='store_const', const=True, default=False, help='When this flag is enabled, transcripts will undergo parsing by spaCy\'s stop words filter and sentence splitter')

    # Collect arguments and execute default function by passing entire
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()