# README for PFM

## PFM is the code base for running the Pathogen Forecast Model (San Diego) collection of regional ocean simulations.

The code here handles all of the automatic running of the PFM. This does not make ROMS executables!!!
---

## Why would you clone this repo?

To port this code to other computers and make devoping easier. And to use python functions for creating various required files for ROMS. E.g., atm_forcing.nc, boundary_condition.nc, etc.


## Significant Changes to PFM

Q_PB & C_PB:
2.2 m3/s, 0.7  t<2025-4-4
2.5 m3/s, 0.5088  2025-4-4 < t < 2025-5-2
2 m3/s, 0.5  t > 2025-5-2

Q_TJ:
for t0 <= t <= 2 Sep 2025
NWM forecast

t >= 2 Sep 2025
uses persistence method. and NWM if rain is detected in NWM Q_TJ

C_TJ
for t < 4 April 2025
0.3                     Q_TJ < 1.83
= .3*1.83 / Q,    Q_TJ > 1.83
wastewater capped at 0.3*1.83 =0.549.

for 4 April 2025 < t < 16 Dec 2025
C0 = 0.65
Cf = 0.045
Qmx = 2.25

16 Dec < t < 14 Jan 2026
same as above, but Qww capped at 5 m3/s

t > 14 Jan 2026
C0 = 0.3
Cf = 0.04
Qmx = 2.25
uses Qww cap at 5 m3/s

Note,
t < 9 Sep 2025
was only putting Q_TJ into 4 of 5 cells, so the total Q_TJ in the model was 80% what we 
intended. But since C_TJ was fixed, Q_WW_TJ was also only 80% what we intended.
t > 9 Sep 2025
fixed.

6/9/2026
Notice that 'river_Vshape' was putting the flow at the bottom, rather than the top. I flipped the array. It is now at the top.
I pushed to PFM. Falk pulled. 
