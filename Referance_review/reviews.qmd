---
title: "Follow-up Research on Two-Phase Transport in PEMFCs"
subtitle: "From Cathode GDL Modeling to Water Management"
format:
  revealjs:
    theme: default
    embed-resources: true
    slide-number: true
    transition: none
---

## Presentation Overview

- Original two-phase transport study
- Attempted FEniCSx implementation
- Difficulties in the modeling process
- Major follow-up studies
- Lessons for future modeling

---

## Original Study

### Chang, Chen, and Teng (2006)

**Effects of two-phase transport in the cathode gas diffusion layer on the performance of a PEMFC**

- Two-phase transport in the cathode GDL
- Liquid-water accumulation
- Oxygen transport through porous media
- Performance loss caused by flooding

---

## Main Physical Mechanism

Water is produced at the cathode catalyst layer.

When liquid water accumulates in the GDL:

- Gas-filled pores are blocked
- Oxygen transport is reduced
- Oxygen concentration near the catalyst layer decreases
- Concentration loss increases
- Fuel-cell performance decreases

---

## Initial Modeling Goal

The original goal was to reproduce the model using FEniCSx.

### Planned work

- Build a two-dimensional GDL domain
- Calculate oxygen concentration
- Calculate liquid-water saturation
- Couple oxygen and water transport
- Extend the original reduced model

---

## Why the Model Was Not Completed

### Main difficulties

- Strong nonlinear coupling
- Saturation-dependent oxygen transport
- Nonlinear capillary-pressure relations
- Unclear interface boundary conditions
- Difficulty maintaining physical saturation values
- Limited information about the numerical procedure

The equations could not be transferred directly into a complete finite-element model.

---

## Follow-up Study 1

### Chen, Chang, and Fang (2007)

**Analysis of water transport in a five-layer model of PEMFC**

The model included:

1. Anode gas diffusion layer
2. Anode catalyst layer
3. Proton exchange membrane
4. Cathode catalyst layer
5. Cathode gas diffusion layer

---

## Five-Layer Water Transport

### Additional phenomena

- Electro-osmotic drag
- Back diffusion
- Membrane water uptake
- Membrane swelling
- Transient water transport

### Main conclusion

Water management is a problem of the entire membrane electrode assembly, not only the cathode GDL.

---

## Follow-up Study 2

### Chen, Chang, and Hsieh (2008)

**Two-phase transport in the cathode gas diffusion layer of PEM fuel cell with a gradient in porosity**

### Main difference

- The original model used uniform GDL porosity
- The follow-up model used a porosity gradient
- Transport properties changed through the GDL thickness
- GDL structure became a design variable

---

## Results of the Porosity-Gradient Study

A suitable porosity gradient:

- Improved liquid-water removal
- Enhanced oxygen transport
- Reduced concentration loss
- Improved predicted fuel-cell performance

The research focus shifted from average material properties to spatial structure.

---

## Later Research Direction

Later studies focused on practical GDL and MPL design.

### Main topics

- PTFE content
- GDL compression
- Microporous-layer composition
- Carbon loading
- Inlet relative humidity
- Double-sided MPL
- Carbon-nanotube-based MPL

---

## Development of the Research

1. Two-phase transport in the cathode GDL
2. Water transport through the full MEA
3. Porosity-gradient GDL design
4. Experimental optimization of GDL and MPL structures

### Overall trend

The research developed from theoretical transport analysis to practical material and structural design.

---

## Lessons for Future Modeling

A more reliable modeling procedure would be:

1. Solve dry oxygen diffusion
2. Verify the oxygen-transport model
3. Solve liquid-water transport separately
4. Couple the two models gradually
5. Introduce spatially varying material properties
6. Add the membrane and catalyst layers later

---

## Conclusion

- The original study investigated flooding in the cathode GDL
- The 2007 study expanded water transport to the full MEA
- The 2008 study introduced a porosity gradient
- Later studies focused on practical GDL and MPL optimization
- Future FEniCSx modeling should proceed through gradual verification

---

## References

::: {.small}

1. M.-H. Chang, F. Chen, and H.-S. Teng,  
   “Effects of two-phase transport in the cathode gas diffusion layer on the performance of a PEMFC,”  
   *Journal of Power Sources*, vol. 160, pp. 268–276, 2006.

2. F. Chen, M.-H. Chang, and C.-F. Fang,  
   “Analysis of water transport in a five-layer model of PEMFC,”  
   *Journal of Power Sources*, vol. 164, pp. 649–658, 2007.

3. F. Chen, M.-H. Chang, and P.-T. Hsieh,  
   “Two-phase transport in the cathode gas diffusion layer of PEM fuel cell with a gradient in porosity,”  
   *International Journal of Hydrogen Energy*, vol. 33, pp. 2525–2529, 2008.


