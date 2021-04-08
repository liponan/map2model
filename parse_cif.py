import glob
import os

from pdb_scraper import download


def main():
    files = glob.glob("pdb_data/cif/*.cif")
    data = []
    for file in files:
        txt = open(file, "r").read()
        for line in txt.splitlines():
            try:
                items = line.rstrip().split()
                if items[0] == "PDB":
                    pdb = items[1]
                if items[0] == "EMDB":
                    emdb = items[1][4:]
                    break
            except IndexError as e:
                print("cannot process", file)
                print(e)
                continue
        print(pdb, emdb)
        data.append(dict(pdb=pdb, emdb=emdb))
    for d in data:
        pdb, emdb = d["pdb"], d["emdb"]
        # Download PDB
        url = "https://files.rcsb.org/download/{}.pdb".format(pdb)
        dest = "pdb_data/pdb/{}.pdb".format(pdb)
        if os.path.isfile(dest):
            print("{} exists, skipping...".format(pdb))
        else:
            print("Fetching {}".format(url))
            try:
                download(url, dest)
            except:
                print("cannot download", url)
        # Download EMDB map
        url = "https://ftp.wwpdb.org/pub/emdb/structures/EMD-{0}/map/emd_{0}.map.gz".format(emdb)
        dest = "pdb_data/map/{}.map.gz".format(pdb)
        if os.path.isfile(dest):
            print("{} exists, skipping...".format(pdb))
        else:
            print("Fetching {}".format(url))
            try:
                download(url, dest)
            except:
                print("cannot download", url)


if __name__ == "__main__":
    main()
