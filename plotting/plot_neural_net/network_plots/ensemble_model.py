import sys
sys.path.append('../')
from pycore.tikzeng import *

# This script does NOT require the calc library or any other file edits.
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    #================================ MODULE 1 (Full Detail) ================================
    to_Conv("m1_input", s_filer=1000, n_filer=1, caption="Input 1",
        to="(-9, 4, 0)", width=1, height=12, depth=1
        ),
    # Branch 1a
    to_Conv("m1_cnn1a", s_filer="", n_filer="", caption="",
        offset="(2.5,1.5,0)", to="(m1_input-east)", width=1, height=8, depth=8
        ),
    to_Pool("m1_pool1a", offset="(0.3,0,0)", to="(m1_cnn1a-east)", width=1, height=8, depth=13),
    to_Conv("m1_cnn2a", s_filer="", n_filer="", caption="",
        offset="(0.3,0,0)", to="(m1_pool1a-east)", width=1, height=8, depth=15
        ),
    to_Pool("m1_pool2a", offset="(0.3,0,0)", to="(m1_cnn2a-east)", width=1, height=8, depth=13),
    to_Conv("m1_cnn3a", s_filer="", n_filer="", caption="",
        offset="(0.3,0,0)", to="(m1_pool2a-east)", width=1, height=8, depth=15
        ),
    # Branch 1b
    to_Conv("m1_cnn1b", s_filer="", n_filer="", caption="",
        offset="(2.5,-1.5,0)", to="(m1_input-east)", width=1, height=8, depth=15
        ),
    to_Pool("m1_pool1b", offset="(0.3,0,0)", to="(m1_cnn1b-east)", width=1, height=8, depth=13),
    to_Conv("m1_cnn2b", s_filer="", n_filer="", caption="",
        offset="(0.3,0,0)", to="(m1_pool1b-east)", width=1, height=8, depth=15
        ),
    to_Pool("m1_pool2b", offset="(0.3,0,0)", to="(m1_cnn2b-east)", width=1, height=8, depth=13),
    to_Conv("m1_cnn3b", s_filer="", n_filer="", caption="",
        offset="(0.3,0,0)", to="(m1_pool2b-east)", width=1, height=8, depth=15
        ),
    # MODIFIED: Module 1 Merge with simple positioning
    r'\node[circle,draw,fill=gray!20,minimum size=15pt,opacity=0.6] (m1_sum) at ([shift={(8,0,0)}]m1_input-east) {$+$};',
    # Module 1 Connections
    r'\draw [connection] ([yshift=2pt]m1_input-east) -- node {\midarrow} (m1_cnn1a-west);',
    r'\draw [connection] ([yshift=-2pt]m1_input-east) -- node {\midarrow} (m1_cnn1b-west);',
    r'\draw [connection] (m1_cnn3a-east) -- (m1_sum);',
    r'\draw [connection] (m1_cnn3b-east) -- (m1_sum);',


    #================================ ELLIPSES for modules 2 to 19 =======================
    r'\node[rotate=90] at (-5,0,0) {\Huge ...};',


    to_Conv("mN_input", s_filer=1000, n_filer=1, caption="Input N",
        to="(-9, -4, 0)", width=1, height=12, depth=1
        ),
    #================================ MODULE N (20) =============================
    # Branch Na
    to_Conv("mN_cnn1a", s_filer="", n_filer="", caption="",
        offset="(2.5,1.5,0)", to="(mN_input-east)", width=1, height=8, depth=8
        ),
    to_Pool("mN_pool1a", offset="(0.3,0,0)", to="(mN_cnn1a-east)", width=1, height=8, depth=13),
    to_Conv("mN_cnn2a", s_filer="", n_filer="", caption="",
        offset="(0.3,0,0)", to="(mN_pool1a-east)", width=1, height=8, depth=15
        ),
    to_Pool("mN_pool2a", offset="(0.3,0,0)", to="(mN_cnn2a-east)", width=1, height=8, depth=13),
    to_Conv("mN_cnn3a", s_filer="", n_filer="", caption="",
        offset="(0.3,0,0)", to="(mN_pool2a-east)", width=1, height=8, depth=15
        ),
    # Branch Nb
    to_Conv("mN_cnn1b", s_filer="", n_filer="", caption="",
        offset="(2.5,-1.5,0)", to="(mN_input-east)", width=1, height=8, depth=15
        ),
    to_Pool("mN_pool1b", offset="(0.3,0,0)", to="(mN_cnn1b-east)", width=1, height=8, depth=13),
    to_Conv("mN_cnn2b", s_filer="", n_filer="", caption="",
        offset="(0.3,0,0)", to="(mN_pool1b-east)", width=1, height=8, depth=15
        ),
    to_Pool("mN_pool2b", offset="(0.3,0,0)", to="(mN_cnn2b-east)", width=1, height=8, depth=13),
    to_Conv("mN_cnn3b", s_filer="", n_filer="", caption="",
        offset="(0.3,0,0)", to="(mN_pool2b-east)", width=1, height=8, depth=15
        ),
    # MODIFIED: Module N Merge with simple positioning
    r'\node[circle,draw,fill=gray!20,minimum size=15pt,opacity=0.6] (mN_sum) at ([shift={(8,0,0)}]mN_input-east) {$+$};',
    # Module N Connections
    r'\draw [connection] ([yshift=2pt]mN_input-east) -- node {\midarrow} (mN_cnn1a-west);',
    r'\draw [connection] (mN_cnn1a-east) -- (mN_pool1a-west);',
    r'\draw [connection] (mN_pool1a-east) -- (mN_cnn2a-west);',
    r'\draw [connection] (mN_cnn2a-east) -- (mN_pool2a-west);',
    r'\draw [connection] (mN_pool2a-east) -- (mN_cnn3a-west);',
    r'\draw [connection] ([yshift=-2pt]mN_input-east) -- node {\midarrow} (mN_cnn1b-west);',
    r'\draw [connection] (mN_cnn1b-east) -- (mN_pool1b-west);',
    r'\draw [connection] (mN_pool1b-east) -- (mN_cnn2b-west);',
    r'\draw [connection] (mN_cnn2b-east) -- (mN_pool2b-west);',
    r'\draw [connection] (mN_pool2b-east) -- (mN_cnn3b-west);',
    r'\draw [connection] (mN_cnn3a-east) -- (mN_sum);',
    r'\draw [connection] (mN_cnn3b-east) -- (mN_sum);',

    #================================ ENSEMBLE MERGE BLOCK =====================================
    # MODIFIED: Ensemble Merge with simple, absolute positioning
    r'\node[circle,draw,fill=gray!20,minimum size=30pt,opacity=0.6] (ensemble_sum) at (4.5,0,0) {$+$};',
    r'\node[below=of ensemble_sum, node distance=0.8cm] {Ensemble Merge};',
    r'\draw [connection] (m1_sum) -- (ensemble_sum);',
    r'\draw [connection] (mN_sum) -- (ensemble_sum);',
    r'\draw [connection] (-0.5,0,0) -- (ensemble_sum);',

    #================================ FINAL MLP & OUTPUT ====================================
    to_SoftMax("mlp1", s_filer=128, caption="MLP Layer-1", offset="(3,0,0)", to="(ensemble_sum)", width=1, height=1, depth=15),
    r'\draw [connection] (ensemble_sum) -- node {\midarrow} (mlp1-west);',
    to_SoftMax("mlp2", s_filer=64, caption="MLP Layer-2", offset="(1.5,0,0)", to="(mlp1-east)", width=1, height=1, depth=15),
    to_connection( "mlp1", "mlp2"),
    to_SoftMax("mlp3", s_filer=32, caption="MLP Layer-3", offset="(1.5,0,0)", to="(mlp2-east)", width=1, height=1, depth=15),
    to_connection( "mlp2", "mlp3"),
    to_Conv("output", s_filer=5, n_filer=1, caption="Output", offset="(1.5,0,0)", to="(mlp3-east)", width=1, depth=1, height=5),
    to_connection( "mlp3", "output"),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
