import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define your neural network architecture
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # Input Vector
    to_Conv("input_vec", s_filer=1000, n_filer=1, caption="Input Vector",
        to="(0,0,0)", width=1, height=20, depth=1      # <-- 3D style change
        ),

    #================================ BRANCH 1 (TOP) ================================
    to_Conv("cnn1a", s_filer=960, n_filer=24, caption="CNN Layer-1a",
        offset="(3,3,0)", to="(input_vec-east)", width=1, height=7, depth=15 # <-- 3D style change
        ),
    to_Pool("pool1a", offset="(1.5,0,0)", to="(cnn1a-east)", width=1, height=7, depth=25, caption="MaxPool"), # <-- 3D style change
    to_Conv("cnn2a", s_filer=200, n_filer=24, caption="CNN Layer-2a",
        offset="(2,0,0)", to="(pool1a-east)", width=1, height=7, depth=30 # <-- 3D style change
        ),
    to_Pool("pool2a", offset="(1.5,0,0)", to="(cnn2a-east)", width=1, height=7, depth=25, caption="MaxPool"), # <-- 3D style change
    to_Conv("cnn3a", s_filer=10, n_filer=24, caption="CNN Layer-3a",
        offset="(1.5,0,0)", to="(pool2a-east)", width=1, height=7, depth=30 # <-- 3D style change
        ),

    #================================ BRANCH 2 (BOTTOM) ===============================
    to_Conv("cnn1b", s_filer=960, n_filer=24, caption="CNN Layer-1b",
        offset="(3,-3,0)", to="(input_vec-east)", width=1, height=7, depth=30 # <-- 3D style change
        ),
    to_Pool("pool1b", offset="(1.5,0,0)", to="(cnn1b-east)", width=1, height=7, depth=25, caption="MaxPool"), # <-- 3D style change
    to_Conv("cnn2b", s_filer=200, n_filer=24, caption="CNN Layer-2b",
        offset="(1.5,0,0)", to="(pool1b-east)", width=1, height=7, depth=30 # <-- 3D style change
        ),
    to_Pool("pool2b", offset="(1.5,0,0)", to="(cnn2b-east)", width=1, height=7, depth=25, caption="MaxPool"), # <-- 3D style change
    to_Conv("cnn3b", s_filer=10, n_filer=24, caption="CNN Layer-3b",
        offset="(1.5,0,0)", to="(pool2b-east)", width=1, height=7, depth=30 # <-- 3D style change
        ),

    #================================ CONNECTIONS =====================================
    to_connection( "input_vec", "cnn1a"),
    to_connection( "cnn1a", "pool1a" ),
    to_connection( "pool1a", "cnn2a"),
    to_connection( "cnn2a", "pool2a" ),
    to_connection( "pool2a", "cnn3a" ),
    to_connection( "input_vec", "cnn1b"),
    to_connection( "cnn1b", "pool1b" ),
    to_connection( "pool1b", "cnn2b"),
    to_connection( "cnn2b", "pool2b" ),
    to_connection( "pool2b", "cnn3b" ),

    #================================ MERGE BLOCK =====================================
    to_Sum("sum1", offset="(1.5,-3,0)", to="(cnn3a-east)", radius=2.5, opacity=0.6),
    to_connection("cnn3a", "sum1"),
    to_connection("cnn3b", "sum1"),

    #================================ MLP & OUTPUT ====================================
    # Note: MLP and Sum layers are 2D elements and do not have a depth parameter.
    to_SoftMax("mlp1", s_filer=24, caption="MLP Layer-1",
        offset="(1.5,0,0)", to="(sum1-east)", width=1, height=1, depth=15
        ),
    to_connection( "sum1", "mlp1"),
    to_SoftMax("mlp2", s_filer=24, caption="MLP Layer-2",
        offset="(1.5,0,0)", to="(mlp1-east)", width=1, height=1, depth=15
        ),
    to_connection( "mlp1", "mlp2"),
    to_Conv("output", s_filer=1, n_filer=2, caption="Output",
        offset="(1.5,0,0)", to="(mlp2-east)", width=1, height=2, depth=1 # <-- 3D style change
        ),
    to_connection( "mlp2", "output"),


    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
