using JuMP
using DataFrames
using CSV
using Dates
using SCIP

players = DataFrame(CSV.File("./data/TEST_SLATE.csv"))