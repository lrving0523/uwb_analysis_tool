# 论文图注（Figure captions）

## Fig. 5 — Geometric Dilution of Precision Along the UAV Trajectory

**(a) Horizontal trajectory and HDOP.** Motion-capture XY trajectory of the UAV colored by
HDOP; the four anchors A0–A3 are marked at the corners of the 0.8 m × 0.8 m landing
platform. Axes are equal-scale.

**(b) Altitude profile and VDOP.** True altitude $z_{GT}$ versus time, colored by VDOP.
The anchor plane is at $z \approx 0.066$ m in the motion-capture frame, so the profile is
shown in absolute mocap coordinates.

**(c) 3D trajectory and PDOP.** Motion-capture XYZ trajectory colored by PDOP, with the
0.8 m platform frame, platform centre (+), anchors A0–A3, and start/end markers.

### Notes

- HDOP, VDOP and PDOP are computed from the **same** geometric Jacobian
  $H_i=(p-a_i)/\lVert p-a_i\rVert$ built from the four true anchor positions and the true
  UAV position: $G=(H^\top H)^{-1}$, $\mathrm{HDOP}=\sqrt{G_{xx}+G_{yy}}$,
  $\mathrm{VDOP}=\sqrt{G_{zz}}$, $\mathrm{PDOP}=\sqrt{\mathrm{tr}(G)}$; therefore
  $\mathrm{PDOP}^2=\mathrm{HDOP}^2+\mathrm{VDOP}^2$ holds for every sample.
- $\mathrm{rank}(H)=3$ for all 607 samples. $\mathrm{cond}(H^\top H)$ has a median of 40
  and a maximum of $1.06\times10^4$ (UAV far from the array and close to the anchor plane);
  the 193 samples with $\mathrm{cond}(H^\top H)>10^3$ are outlined in black.
- All DOP values are finite; no sample was excluded, smoothed or clipped. The color scales
  span the full measured range: HDOP 1.01–3.34 (linear), VDOP 0.65–53.07 (log),
  PDOP 1.53–53.15 (log).
- All three panels use the same 607-sample, 0–60.6 s trajectory segment.
