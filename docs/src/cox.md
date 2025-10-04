## Time-to-event analysis
### Example 7: Cox model under linearity and additivity

The `Qgcomp.jl` module utilizes the Cox proportional hazards models as the underlying model for time-to-event analysis. The interpretation of a `qgcomp_glm_noboot` fit parameter is a conditional (on confounders) hazard ratio for increasing all exposures at once. The qc.survfit1 object demonstrates a time-to- event analysis with qgcompcox.noboot. The default plot is similar to that of `qgcomp_cox_noboot`, in that it yields weights and an overall mixture effect

First, let's return to the data example

```@example advanced
using RData
tf = tempname() * ".RData"
download("https://github.com/alexpkeil1/qgcomp/raw/refs/heads/main/data/metals.RData", tf)
metals = load(tf)["metals"]
println(metals[1:10,:])

# we save the names of the mixture variables in the variable "Xnm"
Xnm = [
    "arsenic","barium","cadmium","calcium","chromium","copper",
    "iron","lead","magnesium","manganese","mercury","selenium","silver",
    "sodium","zinc"

];

covars = ["nitrate","nitrite","sulfate","ph", "total_alkalinity","total_hardness"];

```




```@example advanced
using LSurvival

wu = coxph(@formula(Surv(disease_time, disease_state) ~ iron + lead + arsenic + magnesium + manganese + mercury + selenium + silver + sodium + zinc + mage35), metals);

startvals = vcat(coef(wu)[1:2], 0.0, coef(wu)[3:end])
# fails without some starting values, cadmium is problematic
wu = coxph(@formula(Surv(disease_time, disease_state) ~ iron + lead + cadmium + arsenic + magnesium + manganese + mercury + selenium + silver + sodium + zinc + mage35), metals, start=startvals)

```


```@example advanced
using Qgcomp
qc_survfit1 = qgcomp_cox_noboot(@formula(Surv(disease_time, disease_state) ~ iron + lead + cadmium +  arsenic + magnesium + manganese + mercury + selenium + silver + sodium + zinc + mage35), metals, Xnm, 4)
qc_survfit1
```

```@example advanced
using Plots
weightplot(qc_survfit1)
```