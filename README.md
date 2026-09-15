# Cell Type Constellation viewer


## Overview

This code is prototype of a tool to allow users to represent a cell type
taxonomy that as an interactive constellation plot. The code works by
pre-computing the data necessary to render the interactive visualization,
storing that data in an HDF5 file, and then using the
[cherrypy library](https://cherrypy.dev/) to serve a small app on the user's
local machine within which to visualize the constellation plot.

This code is very much a prototype. Future support is not guaranteed.

That being said, if you have any questions, reach out to
Scott Daniel at scott.daniel@alleninstitute.org

## Installation

This tool is written purely in python. The installation instructions below,
while minimal, assume you have a functioning python environment in which you
want to install this tool and its dependencies.

To install this code, simply clone the repository and then, from the root
directory of the repository run
```
pip install -e .
```

You will know you were successful if you can open a python shell and
successfully run

```
import cell_type_constellations
import cell_type_mapper
```

## Generating the visualization data

The example script
```
examples/serialization_example.py
```
shows how to call a function that reads data from a single h5ad file
and creates an HDF5 file that can be used to create an interactive
constellation plot. The arguments of the function called in that script
are defined in its docstring. See

```
src/cell_type_constellations/serialization/serialization.py
```

## Visualizing the results

This tool comes with a simple web-like app that will allow you to
visualize and configure your constellation plots in any web browser.
The app works by scanning the `app_data/` directory of this repository
and loading any `.h5` files (created by the serialization example
above) for visualization. Just copy all of the constellation plot data
files you want to be able to visualize into that `app_data/` directory and run
```
bash run_visualization.sh
```

you should see output like

```
(constellations) osxltFMD6W:cell_type_constellations scott.daniel$ bash run_visualization.sh 
/Users/scott.daniel/miniconda3/envs/constellations/lib/python3.10/site-packages/cell_type_mapper/taxonomy/utils.py:245: UserWarning: This taxonomy has no mapping from leaf_node -> rows in the cell by gene matrix
  warnings.warn("This taxonomy has no mapping from leaf_node -> rows "
[27/Nov/2024:15:42:24] ENGINE Listening for SIGTERM.
[27/Nov/2024:15:42:24] ENGINE Listening for SIGHUP.
[27/Nov/2024:15:42:24] ENGINE Listening for SIGUSR1.
[27/Nov/2024:15:42:24] ENGINE Bus STARTING
CherryPy Checker:
The Application mounted at '' has an empty config.

[27/Nov/2024:15:42:24] ENGINE Started monitor thread 'Autoreloader'.
[27/Nov/2024:15:42:24] ENGINE Serving on http://127.0.0.1:8080
[27/Nov/2024:15:42:24] ENGINE Bus STARTED
```
appear in your terminal window. The IP address in this line
```
[27/Nov/2024:15:42:24] ENGINE Serving on http://127.0.0.1:8080
```
is the IP address to point your browser towards. You should now be able
to select one of your constellation plots to visualize.
