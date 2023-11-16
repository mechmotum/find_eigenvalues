import bicycleparameters as bp

bicycle = bp.Bicycle("Balanceassist", pathToData="data")
bicycle.add_rider("Jason")
params = bicycle.parameters["Benchmark"]
print(bp.io.remove_uncertainties(params))
