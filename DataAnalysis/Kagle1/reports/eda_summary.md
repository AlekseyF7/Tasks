# EDA Snapshot
- Rows: 9,619
- Price mean/median/std: 19665 / 13000 / 268668
- Year span: 1939 — 2020
- Distance median (km): 125,000

## Price by drivetrain
| Drive   |   count |    mean |   median |
|:--------|--------:|--------:|---------:|
| Front   |    6469 | 20360.8 |    13771 |
| 4x4     |    1993 | 18968.6 |    11000 |
| Rear    |    1157 | 16970.2 |    10976 |

## Top 10 makes by volume & median price
| Make          |   count |   median_price |   median_year |
|:--------------|--------:|---------------:|--------------:|
| HYUNDAI       |    1880 |        18423   |          2014 |
| TOYOTA        |    1834 |        12542.5 |          2012 |
| MERCEDES-BENZ |    1041 |        10976   |          2010 |
| FORD          |     544 |        12231   |          2012 |
| CHEVROLET     |     541 |        12861   |          2014 |
| BMW           |     521 |        14113   |          2011 |
| LEXUS         |     483 |         7840   |          2012 |
| HONDA         |     481 |         9879   |          2011 |
| NISSAN        |     350 |         8000   |          2009 |
| VOLKSWAGEN    |     306 |        11917   |          2012 |

## Numeric correlations vs Price
|                   |        corr |
|:------------------|------------:|
| Year              | -0.0022236  |
| Engine            |  0.00131042 |
| Distance          | -0.00127996 |
| Cylinders         |  0.00076864 |
| Age               |  0.0022236  |
| Distance_per_year | -0.00120405 |