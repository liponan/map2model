import glob
import os
import pandas as pd
import mrcfile


def main():
    files = glob.glob("pdb_data/map/*.map")
    data = []
    for file in files:
        with mrcfile.open(file, "r") as f:
            print(file, end=" ")
            nx = int(f.header.nx)
            ny = int(f.header.ny)
            nz = int(f.header.nz)
            x = float(f.header.cella.x)
            y = float(f.header.cella.y)
            z = float(f.header.cella.z)
            dx = x / nx
            dy = y / ny
            dz = z / nz
            if abs(dx - dy) + abs(dy - dz) + abs(dx - dz) > 0.01:
                print("resolution inconsistent {:.5f} {:.5f} {:.5f}".format(dx, dy, dz))
                continue
            else:
                print("nx {:3d} ny {:3d} nz {:3d} x {:5.1f} y {:5.1f} z {:5.1f} dx {:5.3f} dy {:5.3f} dz {:5.3f}".format(nx, ny, nz, x, y, z, dx, dy, dz))
        data.append(
            dict(nx=nx, ny=ny, nz=nz, max_n=max(nx, ny, nz), x=x, y=y, z=z, max_dim=max(x, y, z),
                 dx=dx, dy=dy, dz=dz))
    df = pd.DataFrame(data)
    print(df)
    df.to_csv("pdb_data/map_summay.csv")


if __name__ == "__main__":
    main()
