# Propagation

Propagation is handled separately from primitive accumulation.

This separation matters because changes to the custom kernel backend do not automatically change the propagation model.

## Main propagation class

The primary class is `ASM_parallel` in `hsplat/propagations.py`.

Its public path is:

1. `forward(wavefront, z, linear_conv=True, phase_compensation=None)`
2. `compute_H(...)`
3. `prop(...)`

## `forward(...)`

`forward(...)` is the high-level entry point.

It:

- skips work when `z` is effectively zero and no phase compensation is needed
- computes the propagation transfer function `H`
- applies propagation through `prop(...)`

## `compute_H(...)`

This function creates the propagation transfer function in frequency space.

Key steps:

- determine output resolution and optional linear-convolution padding
- sample the frequency axes `fx` and `fy`
- create the aperture filter `H_filter`
- compute the propagation phase term from the angular spectrum relation

When `prop_dist` is near zero, the function returns just the aperture filter.

## `prop(...)`

This function performs the actual FFT-domain propagation.

Its core flow is:

1. optionally pad the input for linear convolution
2. FFT the input wavefront
3. multiply by `H`
4. optionally apply `phase_compensation`
5. inverse FFT back to the spatial domain
6. crop back to the requested output resolution

## Where propagation is used

Propagation appears in multiple places:

- point rendering in `cgh_from_point(...)`
- silhouette-based composition
- alpha-wave-blending composition
- focal-slice generation after the main wavefront is produced

That means propagation is both:

- part of the rendering method implementation
- and part of the downstream visualization/output workflow
