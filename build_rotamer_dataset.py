import os
import argparse
from m2m_utils import list_files, parse_h5_attrs, parse_pdb_id


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5_path", type=str, help="Where to find map h5 files")
    p.add_argument("output_path", type=str, default="./", help="Where to put parsed h5 files")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    h5_files = list_files(args.h5_path)
    for h5_file in h5_files:
        pdb_id = parse_pdb_id(h5_file)
        print("{} => {}".format(h5_file, pdb_id))
        try:
            df = parse_h5_attrs(h5_file)
            df.to_csv(os.path.join(args.output_path, "{}.h5".format(pdb_id)))
        except:
            print("cannot process {}".format(h5_file))


if __name__ == "__main__":
    main()