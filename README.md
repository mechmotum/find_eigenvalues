# Balance-assist bicycle weave mode identification

- Raw files from balance-assist bicycle can be found in `eigenvalues_balance_assist_120523`. 
- `ReadDataMarten.m` decodes the files and writes `.mat` files to `finalized_logs`.
- `PlotData` is used to plot the data, extract relevant segments and write these to the `finalized_logs`directory.
- `AnalyzeData.py` is used to fit a function to the data in order to determine the eigenvalues, and to plot these eigenvalues.

A latex project and document describing the experiments and the results can be found in the `write-up` directory. 

