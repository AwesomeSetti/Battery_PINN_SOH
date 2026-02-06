# Fast Formation Data

Andrew Weng

9/22/2021

DOI: https://doi.org/10.7302/pa3f-4w30


## Description

The dataset supports research to investigate the impact of fast formation protocols on battery lifetime. The dataset has been used to explore data-driven approaches in battery lifetime estimation. 

Source code used to generate the results for this work has been included.


## Methodology

Forty prismatic lithium ion pouch cells were built at the University of Michigan Battery Laboratory. The cells have a nominal capacity of 2.36Ah and comprise of a NCM111 cathode and graphite anode.

Cells were formed using two different formation protocols: "fast formation" and "baseline formation".

After formation, cells were put under cycle life testing at room temperature and 45degC. Cells were cycled until the discharge capacities dropped below 50% of the initial capacities.

Data was collected by the cycler equipment (Maccor) during both the formation process as well as during the cycling test. Data was processed in the Voltaiq software and subsequently exported as .csv files.

## Description of Folders

- `code` contains a copy of the source code (`https://doi.org/10.5281/zenodo.5525258`)
- `data` includes all of the raw cycler data files, including during formation, during cycling, and for the coin cells
- `documents` contain cell tracker files and test schedules
- `output` contains post-processed data outputs

