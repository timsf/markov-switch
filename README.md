# markov-switch

Replication code for "Exact Bayesian Inference for Markov Swithing Diffusions", open access at https://arxiv.org/abs/2502.09126.

    .
    ├── markov_switch    # Code root
    └── paper            # Materials pertaining to the academic paper


## Installation Instructions for Ubuntu/OSX

0. Install the generic dependencies `Python 3.13`, `uv`, `git`.

1. Define your project root (`[project]`) and navigate there:

    ```shell
    mkdir [project]
    cd [project]
    ```

2. Clone the repository:

    ```shell
    git clone https://github.com/timsf/markov-switch.git
    ```

3. Start the `Jupyter` server:

    ```shell
    uv run jupyter notebook
    ```

4. Access the `Jupyter` server in your browser and navigate to the notebook of interest.


## Reproducing Figures in the Paper

To reproduce Figures 1, 3, 4, 5, 6, first run `f109_mcmc.ipynb`, `f109_mcem.ipynb`, `f109_approx.ipynb`, `f109_impute4.ipynb` to generate algorithm output, then `f109_plots.ipynb` to draw the plots.

To reproduce Figures 7 and 8, first run `simstud_base.ipynb`, `simstud_extend.ipynb`, `simstud_infill.ipynb` to generate algorithm output, then `simstud_plots.ipynb` to draw the plots.


## Reference

    @article{stumpf2025exact,
        title={Exact Bayesian inference for Markov switching diffusions},
        author={Stumpf-F{\'e}tizon, Timoth{\'e}e and {\L}atuszy{\'n}ski, Krzysztof and Palczewski, Jan and Roberts, Gareth},
        journal={arXiv preprint arXiv:2502.09126},
        year={2025}
    }
