# Sources and indicator notes

All historical indicator observations are requested from the World Bank Indicators API:
https://api.worldbank.org/v2/

The app requests `current year - 70` through `current year - 1`, inclusive, and displays the actual non-missing coverage returned by the API. The API may have gaps or no values for a particular country/indicator/year. No missing values are fabricated.

| Display name | World Bank indicator code | Units shown | Definition note |
|---|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | years | Expected years lived under current age-specific mortality patterns. |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | % of labor force | National-estimate unemployment rate. |
| GDP per capita | `NY.GDP.PCAP.CD` | current US$ per person | GDP divided by midyear population; not average wealth or household income. |
| Crude birth rate | `SP.DYN.CBRT.IN` | births per 1,000 people per year | A rate, not a count of births. |
| Net migration | `SM.POP.NETM` | people | Net migration, not a count of emigrants. |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | per 100,000 people | Intentional homicides per 100,000 people. |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | % annual change | Annual percentage change in consumer prices. |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | % of GDP | Central government debt as a percentage of GDP. |
| Total reserves including gold | `FI.RES.TOTL.CD` | current US$ | Total reserves including gold. |
| External balance on goods/services | `NE.RSB.GNFS.ZS` | % of GDP | External balance on goods and services as a percentage of GDP. |

Country codes used by the API: Mexico `MEX`, Brazil `BRA`, Argentina `ARG`, Colombia `COL`, Chile `CHL`, Peru `PER`, Venezuela `VEN`, Ecuador `ECU`, and Uruguay `URY`.

The app shows the exact API endpoint and retrieval date for the selected country/indicator. Indicator definitions and availability may be revised by the World Bank; use the linked API metadata for project citations.
