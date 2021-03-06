# DiscoPoP graph analyzer
DiscoPoP profiler is accompanied by a Python framework, specifically designed to analyze the profiler output files, generate a CU graph, detect potential parallel patterns, and suggest OpenMP parallelizations.
Currently, the following five patterns can be detected:
* Reduction
* Do-All
* Pipeline
* Geometric Decomposition
* Task Parallelism

## Getting started
We assume that you have already run the DiscoPoP profiler on the target sequential application, and the following files are created in the current working directory:
* `Data.xml` (CU information in XML format created by *CUGeneration* pass)
* `<app_name>_dep.txt` (Data dependences created by *DPInstrumentation* pass)
* `reduction.txt` and `loop_counter_output.txt` (Reduction operations and loop iteration data identified by *DPReduction* pass)

In case any of the files mentioned above are missing, please follow the [DiscoPoP manual](../README.md) to generate them.

### Pre-requisites
To use the graph analyzer tool, you need to have Python 3.6+ installed on your system. Further Python dependencies can be installed using the following command:
`pip install -r requirements.txt`

### Usage
To run the graph analyzer, you can use the following command:

`python3 -m discopop_explorer --path <path-to-your-output>`

You can specify the path to DiscoPoP output files. Then, the Python script searches within this path to find the required files. Nevertheless, if you are interested in passing a specific location to each file, here is the detailed usage:

    `discopop_explorer [--path <path>] [--cu-xml <cuxml>] [--dep-file <depfile>] [--plugins <plugs>] [--loop-counter <loopcount>] [--reduction <reduction>] [--json <json>]`

Options:
```
    --path=<path>               Directory with input data [default: ./]
    --cu-xml=<cuxml>            CU node xml file [default: Data.xml].
    --dep-file=<depfile>        Dependencies text file [default: dep.txt].
    --loop-counter=<loopcount>  Loop counter data [default: loop_counter_output.txt].
    --reduction=<reduction>     Reduction variables file [default: reduction.txt].
    --plugins=<plugs>           Plugins to execute
    --json                      Output result as a json file to specified path
    -h --help                   Show this screen.
    --version                   Show version.
```

By default, running the graph analyzer will print out the list of patterns along with OpenMP parallelization suggestions to the standard output. You can also obtain the results in JSON format by passing `--json` argument to the Python script.

### Walkthrough example
The **test/** folder contains a number of precomputed inputs for testing the tool, e.g., *atax* from Polybench benchmark suite.
You can try out this example workflow.

**test/reduction/** contains source code and precomputed DiscoPoP output for a simple reduction loop.
The loop itself sums up all numbers from 1 to n.

You can run DiscoPoP on **main.c** or just use included output.

After that, you can run **discopop_explorer**. The **--path** argument should point to the output of the DiscoPoP.

In this example, the output for reduction will point to the lines 6-9, and it will suggest **pragma omp parallel for** OpenMP directive for parallelizing the loop.
You will also find **i** classified as a private variable and **sum** as a reduction variable. Thus, the parallelization directive would be suggested as follows:

```#pragma omp parallel for private(i) reduction(+:sum)```

The suggested pattern is demonstrated in **mainp.c**
